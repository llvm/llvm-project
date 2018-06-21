//===-- hwasan.cc -----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// HWAddressSanitizer runtime.
//===----------------------------------------------------------------------===//

#include "hwasan.h"
#include "hwasan_thread.h"
#include "hwasan_poisoning.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "ubsan/ubsan_flags.h"
#include "ubsan/ubsan_init.h"

// ACHTUNG! No system header includes in this file.

using namespace __sanitizer;

namespace __hwasan {

void EnterSymbolizer() {
  HwasanThread *t = GetCurrentThread();
  CHECK(t);
  t->EnterSymbolizer();
}
void ExitSymbolizer() {
  HwasanThread *t = GetCurrentThread();
  CHECK(t);
  t->LeaveSymbolizer();
}
bool IsInSymbolizer() {
  HwasanThread *t = GetCurrentThread();
  return t && t->InSymbolizer();
}

static Flags hwasan_flags;

Flags *flags() {
  return &hwasan_flags;
}

int hwasan_inited = 0;
bool hwasan_init_is_running;

int hwasan_report_count = 0;

void Flags::SetDefaults() {
#define HWASAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "hwasan_flags.inc"
#undef HWASAN_FLAG
}

static void RegisterHwasanFlags(FlagParser *parser, Flags *f) {
#define HWASAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "hwasan_flags.inc"
#undef HWASAN_FLAG
}

static void InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.external_symbolizer_path = GetEnv("HWASAN_SYMBOLIZER_PATH");
    cf.malloc_context_size = 20;
    cf.handle_ioctl = true;
    // FIXME: test and enable.
    cf.check_printf = false;
    cf.intercept_tls_get_addr = true;
    cf.exitcode = 99;
    cf.handle_sigill = kHandleSignalExclusive;
    OverrideCommonFlags(cf);
  }

  Flags *f = flags();
  f->SetDefaults();

  FlagParser parser;
  RegisterHwasanFlags(&parser, f);
  RegisterCommonFlags(&parser);

#if HWASAN_CONTAINS_UBSAN
  __ubsan::Flags *uf = __ubsan::flags();
  uf->SetDefaults();

  FlagParser ubsan_parser;
  __ubsan::RegisterUbsanFlags(&ubsan_parser, uf);
  RegisterCommonFlags(&ubsan_parser);
#endif

  // Override from user-specified string.
  if (__hwasan_default_options)
    parser.ParseString(__hwasan_default_options());
#if HWASAN_CONTAINS_UBSAN
  const char *ubsan_default_options = __ubsan::MaybeCallUbsanDefaultOptions();
  ubsan_parser.ParseString(ubsan_default_options);
#endif

  const char *hwasan_options = GetEnv("HWASAN_OPTIONS");
  parser.ParseString(hwasan_options);
#if HWASAN_CONTAINS_UBSAN
  ubsan_parser.ParseString(GetEnv("UBSAN_OPTIONS"));
#endif
  VPrintf(1, "HWASAN_OPTIONS: %s\n", hwasan_options ? hwasan_options : "<empty>");

  InitializeCommonFlags();

  if (Verbosity()) ReportUnrecognizedFlags();

  if (common_flags()->help) parser.PrintFlagDescriptions();
}

void GetStackTrace(BufferedStackTrace *stack, uptr max_s, uptr pc, uptr bp,
                   void *context, bool request_fast_unwind) {
  HwasanThread *t = GetCurrentThread();
  if (!t || !StackTrace::WillUseFastUnwind(request_fast_unwind)) {
    // Block reports from our interceptors during _Unwind_Backtrace.
    SymbolizerScope sym_scope;
    return stack->Unwind(max_s, pc, bp, context, 0, 0, request_fast_unwind);
  }
  stack->Unwind(max_s, pc, bp, context, t->stack_top(), t->stack_bottom(),
                request_fast_unwind);
}

void PrintWarning(uptr pc, uptr bp) {
  GET_FATAL_STACK_TRACE_PC_BP(pc, bp);
  ReportInvalidAccess(&stack, 0);
}

} // namespace __hwasan

// Interface.

using namespace __hwasan;

void __hwasan_init() {
  CHECK(!hwasan_init_is_running);
  if (hwasan_inited) return;
  hwasan_init_is_running = 1;
  SanitizerToolName = "HWAddressSanitizer";

  InitTlsSize();

  CacheBinaryName();
  InitializeFlags();

  __sanitizer_set_report_path(common_flags()->log_path);

  InitializeInterceptors();
  InstallDeadlySignalHandlers(HwasanOnDeadlySignal);
  InstallAtExitHandler(); // Needs __cxa_atexit interceptor.

  DisableCoreDumperIfNecessary();
  if (!InitShadow()) {
    Printf("FATAL: HWAddressSanitizer can not mmap the shadow memory.\n");
    Printf("FATAL: Make sure to compile with -fPIE and to link with -pie.\n");
    Printf("FATAL: Disabling ASLR is known to cause this error.\n");
    Printf("FATAL: If running under GDB, try "
           "'set disable-randomization off'.\n");
    DumpProcessMap();
    Die();
  }

  Symbolizer::GetOrInit()->AddHooks(EnterSymbolizer, ExitSymbolizer);

  InitializeCoverage(common_flags()->coverage, common_flags()->coverage_dir);

  HwasanTSDInit(HwasanTSDDtor);

  HwasanAllocatorInit();

  HwasanThread *main_thread = HwasanThread::Create(nullptr, nullptr);
  SetCurrentThread(main_thread);
  main_thread->ThreadStart();

#if HWASAN_CONTAINS_UBSAN
  __ubsan::InitAsPlugin();
#endif

  VPrintf(1, "HWAddressSanitizer init done\n");

  hwasan_init_is_running = 0;
  hwasan_inited = 1;
}

void __hwasan_print_shadow(const void *x, uptr size) {
  // FIXME:
  Printf("FIXME: __hwasan_print_shadow unimplemented\n");
}

sptr __hwasan_test_shadow(const void *p, uptr sz) {
  if (sz == 0)
    return -1;
  tag_t ptr_tag = GetTagFromPointer((uptr)p);
  if (ptr_tag == 0)
    return -1;
  uptr ptr_raw = GetAddressFromPointer((uptr)p);
  uptr shadow_first = MEM_TO_SHADOW(ptr_raw);
  uptr shadow_last = MEM_TO_SHADOW(ptr_raw + sz - 1);
  for (uptr s = shadow_first; s <= shadow_last; ++s)
    if (*(tag_t*)s != ptr_tag)
      return SHADOW_TO_MEM(s) - ptr_raw;
  return -1;
}

u16 __sanitizer_unaligned_load16(const uu16 *p) {
  return *p;
}
u32 __sanitizer_unaligned_load32(const uu32 *p) {
  return *p;
}
u64 __sanitizer_unaligned_load64(const uu64 *p) {
  return *p;
}
void __sanitizer_unaligned_store16(uu16 *p, u16 x) {
  *p = x;
}
void __sanitizer_unaligned_store32(uu32 *p, u32 x) {
  *p = x;
}
void __sanitizer_unaligned_store64(uu64 *p, u64 x) {
  *p = x;
}

template<unsigned X>
__attribute__((always_inline))
static void SigIll() {
#if defined(__aarch64__)
  asm("hlt %0\n\t" ::"n"(X));
#elif defined(__x86_64__) || defined(__i386__)
  asm("ud2\n\t");
#else
  // FIXME: not always sigill.
  __builtin_trap();
#endif
  // __builtin_unreachable();
}

enum class ErrorAction { Abort, Recover };
enum class AccessType { Load, Store };

template <ErrorAction EA, AccessType AT, unsigned LogSize>
__attribute__((always_inline, nodebug)) static void CheckAddress(uptr p) {
  tag_t ptr_tag = GetTagFromPointer(p);
  uptr ptr_raw = p & ~kAddressTagMask;
  tag_t mem_tag = *(tag_t *)MEM_TO_SHADOW(ptr_raw);
  if (UNLIKELY(ptr_tag != mem_tag)) {
    SigIll<0x100 + 0x20 * (EA == ErrorAction::Recover) +
           0x10 * (AT == AccessType::Store) + LogSize>();
    if (EA == ErrorAction::Abort) __builtin_unreachable();
  }
}

template <ErrorAction EA, AccessType AT>
__attribute__((always_inline, nodebug)) static void CheckAddressSized(uptr p,
                                                                      uptr sz) {
  CHECK_NE(0, sz);
  tag_t ptr_tag = GetTagFromPointer(p);
  uptr ptr_raw = p & ~kAddressTagMask;
  tag_t *shadow_first = (tag_t *)MEM_TO_SHADOW(ptr_raw);
  tag_t *shadow_last = (tag_t *)MEM_TO_SHADOW(ptr_raw + sz - 1);
  for (tag_t *t = shadow_first; t <= shadow_last; ++t)
    if (UNLIKELY(ptr_tag != *t)) {
      SigIll<0x100 + 0x20 * (EA == ErrorAction::Recover) +
             0x10 * (AT == AccessType::Store) + 0xf>();
      if (EA == ErrorAction::Abort) __builtin_unreachable();
    }
}

void __hwasan_load(uptr p, uptr sz) {
  CheckAddressSized<ErrorAction::Abort, AccessType::Load>(p, sz);
}
void __hwasan_load1(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Load, 0>(p);
}
void __hwasan_load2(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Load, 1>(p);
}
void __hwasan_load4(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Load, 2>(p);
}
void __hwasan_load8(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Load, 3>(p);
}
void __hwasan_load16(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Load, 4>(p);
}

void __hwasan_load_noabort(uptr p, uptr sz) {
  CheckAddressSized<ErrorAction::Recover, AccessType::Load>(p, sz);
}
void __hwasan_load1_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Load, 0>(p);
}
void __hwasan_load2_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Load, 1>(p);
}
void __hwasan_load4_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Load, 2>(p);
}
void __hwasan_load8_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Load, 3>(p);
}
void __hwasan_load16_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Load, 4>(p);
}

void __hwasan_store(uptr p, uptr sz) {
  CheckAddressSized<ErrorAction::Abort, AccessType::Store>(p, sz);
}
void __hwasan_store1(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Store, 0>(p);
}
void __hwasan_store2(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Store, 1>(p);
}
void __hwasan_store4(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Store, 2>(p);
}
void __hwasan_store8(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Store, 3>(p);
}
void __hwasan_store16(uptr p) {
  CheckAddress<ErrorAction::Abort, AccessType::Store, 4>(p);
}

void __hwasan_store_noabort(uptr p, uptr sz) {
  CheckAddressSized<ErrorAction::Recover, AccessType::Store>(p, sz);
}
void __hwasan_store1_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Store, 0>(p);
}
void __hwasan_store2_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Store, 1>(p);
}
void __hwasan_store4_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Store, 2>(p);
}
void __hwasan_store8_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Store, 3>(p);
}
void __hwasan_store16_noabort(uptr p) {
  CheckAddress<ErrorAction::Recover, AccessType::Store, 4>(p);
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char* __hwasan_default_options() { return ""; }
}  // extern "C"
#endif

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  GET_FATAL_STACK_TRACE_PC_BP(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME());
  stack.Print();
}
} // extern "C"
