//===-- dfsan.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// DataFlowSanitizer runtime.  This file defines the public interface to
// DataFlowSanitizer as well as the definition of certain runtime functions
// called automatically by the compiler (specifically the instrumentation pass
// in llvm/lib/Transforms/Instrumentation/DataFlowSanitizer.cpp).
//
// The public interface is defined in include/sanitizer/dfsan_interface.h whose
// functions are prefixed dfsan_ while the compiler interface functions are
// prefixed __dfsan_.
//===----------------------------------------------------------------------===//

#include "dfsan/dfsan.h"

#include "dfsan/dfsan_chained_origin_depot.h"
#include "dfsan/dfsan_flags.h"
#include "dfsan/dfsan_origin.h"
#include "dfsan/dfsan_thread.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __dfsan;

typedef atomic_uint16_t atomic_dfsan_label;
static const dfsan_label kInitializingLabel = -1;

static const uptr kNumLabels = 1 << (sizeof(dfsan_label) * 8);

static atomic_dfsan_label __dfsan_last_label;
static dfsan_label_info __dfsan_label_info[kNumLabels];

Flags __dfsan::flags_data;

// The size of TLS variables. These constants must be kept in sync with the ones
// in DataFlowSanitizer.cpp.
static const int kDFsanArgTlsSize = 800;
static const int kDFsanRetvalTlsSize = 800;

SANITIZER_INTERFACE_ATTRIBUTE THREADLOCAL u64
    __dfsan_retval_tls[kDFsanRetvalTlsSize / sizeof(u64)];
SANITIZER_INTERFACE_ATTRIBUTE THREADLOCAL u64
    __dfsan_arg_tls[kDFsanArgTlsSize / sizeof(u64)];

SANITIZER_INTERFACE_ATTRIBUTE uptr __dfsan_shadow_ptr_mask;

// On Linux/x86_64, memory is laid out as follows:
//
// +--------------------+ 0x800000000000 (top of memory)
// | application memory |
// +--------------------+ 0x700000008000 (kAppAddr)
// |                    |
// |       unused       |
// |                    |
// +--------------------+ 0x300200000000 (kUnusedAddr)
// |    union table     |
// +--------------------+ 0x300000000000 (kUnionTableAddr)
// |       origin       |
// +--------------------+ 0x200000000000 (kOriginAddr)
// |   shadow memory    |
// +--------------------+ 0x000000010000 (kShadowAddr)
// | reserved by kernel |
// +--------------------+ 0x000000000000
//
// To derive a shadow memory address from an application memory address,
// bits 44-46 are cleared to bring the address into the range
// [0x000000008000,0x100000000000).  Then the address is shifted left by 1 to
// account for the double byte representation of shadow labels and move the
// address into the shadow memory range.  See the function shadow_for below.

// On Linux/MIPS64, memory is laid out as follows:
//
// +--------------------+ 0x10000000000 (top of memory)
// | application memory |
// +--------------------+ 0xF000008000 (kAppAddr)
// |                    |
// |       unused       |
// |                    |
// +--------------------+ 0x2200000000 (kUnusedAddr)
// |    union table     |
// +--------------------+ 0x2000000000 (kUnionTableAddr)
// |   shadow memory    |
// +--------------------+ 0x0000010000 (kShadowAddr)
// | reserved by kernel |
// +--------------------+ 0x0000000000

// On Linux/AArch64 (39-bit VMA), memory is laid out as follow:
//
// +--------------------+ 0x8000000000 (top of memory)
// | application memory |
// +--------------------+ 0x7000008000 (kAppAddr)
// |                    |
// |       unused       |
// |                    |
// +--------------------+ 0x1200000000 (kUnusedAddr)
// |    union table     |
// +--------------------+ 0x1000000000 (kUnionTableAddr)
// |   shadow memory    |
// +--------------------+ 0x0000010000 (kShadowAddr)
// | reserved by kernel |
// +--------------------+ 0x0000000000

// On Linux/AArch64 (42-bit VMA), memory is laid out as follow:
//
// +--------------------+ 0x40000000000 (top of memory)
// | application memory |
// +--------------------+ 0x3ff00008000 (kAppAddr)
// |                    |
// |       unused       |
// |                    |
// +--------------------+ 0x1200000000 (kUnusedAddr)
// |    union table     |
// +--------------------+ 0x8000000000 (kUnionTableAddr)
// |   shadow memory    |
// +--------------------+ 0x0000010000 (kShadowAddr)
// | reserved by kernel |
// +--------------------+ 0x0000000000

// On Linux/AArch64 (48-bit VMA), memory is laid out as follow:
//
// +--------------------+ 0x1000000000000 (top of memory)
// | application memory |
// +--------------------+ 0xffff00008000 (kAppAddr)
// |       unused       |
// +--------------------+ 0xaaaab0000000 (top of PIE address)
// | application PIE    |
// +--------------------+ 0xaaaaa0000000 (top of PIE address)
// |                    |
// |       unused       |
// |                    |
// +--------------------+ 0x1200000000 (kUnusedAddr)
// |    union table     |
// +--------------------+ 0x8000000000 (kUnionTableAddr)
// |   shadow memory    |
// +--------------------+ 0x0000010000 (kShadowAddr)
// | reserved by kernel |
// +--------------------+ 0x0000000000

typedef atomic_dfsan_label dfsan_union_table_t[kNumLabels][kNumLabels];

#ifdef DFSAN_RUNTIME_VMA
// Runtime detected VMA size.
int __dfsan::vmaSize;
#endif

static uptr UnusedAddr() {
  return UnionTableAddr() + sizeof(dfsan_union_table_t);
}

static atomic_dfsan_label *union_table(dfsan_label l1, dfsan_label l2) {
  return &(*(dfsan_union_table_t *) UnionTableAddr())[l1][l2];
}

// Checks we do not run out of labels.
static void dfsan_check_label(dfsan_label label) {
  if (label == kInitializingLabel) {
    Report("FATAL: DataFlowSanitizer: out of labels\n");
    Die();
  }
}

// Resolves the union of two unequal labels.  Nonequality is a precondition for
// this function (the instrumentation pass inlines the equality test).
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
dfsan_label __dfsan_union(dfsan_label l1, dfsan_label l2) {
  DCHECK_NE(l1, l2);

  if (l1 == 0)
    return l2;
  if (l2 == 0)
    return l1;

  // If no labels have been created, yet l1 and l2 are non-zero, we are using
  // fast16labels mode.
  if (atomic_load(&__dfsan_last_label, memory_order_relaxed) == 0)
    return l1 | l2;

  if (l1 > l2)
    Swap(l1, l2);

  atomic_dfsan_label *table_ent = union_table(l1, l2);
  // We need to deal with the case where two threads concurrently request
  // a union of the same pair of labels.  If the table entry is uninitialized,
  // (i.e. 0) use a compare-exchange to set the entry to kInitializingLabel
  // (i.e. -1) to mark that we are initializing it.
  dfsan_label label = 0;
  if (atomic_compare_exchange_strong(table_ent, &label, kInitializingLabel,
                                     memory_order_acquire)) {
    // Check whether l2 subsumes l1.  We don't need to check whether l1
    // subsumes l2 because we are guaranteed here that l1 < l2, and (at least
    // in the cases we are interested in) a label may only subsume labels
    // created earlier (i.e. with a lower numerical value).
    if (__dfsan_label_info[l2].l1 == l1 ||
        __dfsan_label_info[l2].l2 == l1) {
      label = l2;
    } else {
      label =
        atomic_fetch_add(&__dfsan_last_label, 1, memory_order_relaxed) + 1;
      dfsan_check_label(label);
      __dfsan_label_info[label].l1 = l1;
      __dfsan_label_info[label].l2 = l2;
    }
    atomic_store(table_ent, label, memory_order_release);
  } else if (label == kInitializingLabel) {
    // Another thread is initializing the entry.  Wait until it is finished.
    do {
      internal_sched_yield();
      label = atomic_load(table_ent, memory_order_acquire);
    } while (label == kInitializingLabel);
  }
  return label;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
dfsan_label __dfsan_union_load(const dfsan_label *ls, uptr n) {
  dfsan_label label = ls[0];
  for (uptr i = 1; i != n; ++i) {
    dfsan_label next_label = ls[i];
    if (label != next_label)
      label = __dfsan_union(label, next_label);
  }
  return label;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
dfsan_label __dfsan_union_load_fast16labels(const dfsan_label *ls, uptr n) {
  dfsan_label label = ls[0];
  for (uptr i = 1; i != n; ++i)
    label |= ls[i];
  return label;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
void __dfsan_unimplemented(char *fname) {
  if (flags().warn_unimplemented)
    Report("WARNING: DataFlowSanitizer: call to uninstrumented function %s\n",
           fname);
}

// Use '-mllvm -dfsan-debug-nonzero-labels' and break on this function
// to try to figure out where labels are being introduced in a nominally
// label-free program.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dfsan_nonzero_label() {
  if (flags().warn_nonzero_labels)
    Report("WARNING: DataFlowSanitizer: saw nonzero label\n");
}

// Indirect call to an uninstrumented vararg function. We don't have a way of
// handling these at the moment.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__dfsan_vararg_wrapper(const char *fname) {
  Report("FATAL: DataFlowSanitizer: unsupported indirect call to vararg "
         "function %s\n", fname);
  Die();
}

// Like __dfsan_union, but for use from the client or custom functions.  Hence
// the equality comparison is done here before calling __dfsan_union.
SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
dfsan_union(dfsan_label l1, dfsan_label l2) {
  if (l1 == l2)
    return l1;
  return __dfsan_union(l1, l2);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
dfsan_label dfsan_create_label(const char *desc, void *userdata) {
  dfsan_label label =
      atomic_fetch_add(&__dfsan_last_label, 1, memory_order_relaxed) + 1;
  dfsan_check_label(label);
  __dfsan_label_info[label].l1 = __dfsan_label_info[label].l2 = 0;
  __dfsan_label_info[label].desc = desc;
  __dfsan_label_info[label].userdata = userdata;
  return label;
}

// For platforms which support slow unwinder only, we need to restrict the store
// context size to 1, basically only storing the current pc, because the slow
// unwinder which is based on libunwind is not async signal safe and causes
// random freezes in forking applications as well as in signal handlers.
// DFSan supports only Linux. So we do not restrict the store context size.
#define GET_STORE_STACK_TRACE_PC_BP(pc, bp) \
  BufferedStackTrace stack;                 \
  stack.Unwind(pc, bp, nullptr, true, flags().store_context_size);

#define PRINT_CALLER_STACK_TRACE        \
  {                                     \
    GET_CALLER_PC_BP_SP;                \
    (void)sp;                           \
    GET_STORE_STACK_TRACE_PC_BP(pc, bp) \
    stack.Print();                      \
  }

/*
static u32 ChainOrigin(u32 id, StackTrace *stack, bool from_init = false) {
  // StackDepot is not async signal safe. Do not create new chains in a signal
  // handler.
  DFsanThread *t = GetCurrentThread();
  if (t && t->InSignalHandler())
    return id;

  // As an optimization the origin of an application byte is updated only when
  // its shadow is non-zero. Because we are only interested in the origins of
  // taint labels, it does not matter what origin a zero label has. This reduces
  // memory write cost. MSan does similar optimization. The following invariant
  // may not hold because of some bugs. We check the invariant to help debug.
  if (!from_init && id == 0 && flags().check_origin_invariant) {
    Printf("  DFSan found invalid origin invariant\n");
    PRINT_CALLER_STACK_TRACE
  }

  Origin o = Origin::FromRawId(id);
  stack->tag = StackTrace::TAG_UNKNOWN;
  Origin chained = Origin::CreateChainedOrigin(o, stack);
  return chained.raw_id();
}
*/

static void WriteShadowIfDifferent(dfsan_label label, uptr shadow_addr,
                                   uptr size) {
  dfsan_label *labelp = (dfsan_label *)shadow_addr;
  for (; size != 0; --size, ++labelp) {
    // Don't write the label if it is already the value we need it to be.
    // In a program where most addresses are not labeled, it is common that
    // a page of shadow memory is entirely zeroed.  The Linux copy-on-write
    // implementation will share all of the zeroed pages, making a copy of a
    // page when any value is written.  The un-sharing will happen even if
    // the value written does not change the value in memory.  Avoiding the
    // write when both |label| and |*labelp| are zero dramatically reduces
    // the amount of real memory used by large programs.
    if (label == *labelp)
      continue;

    *labelp = label;
  }
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dfsan_set_label(
    dfsan_label label, void *addr, uptr size) {
  const uptr beg_shadow_addr = (uptr)__dfsan::shadow_for(addr);

  if (0 != label) {
    WriteShadowIfDifferent(label, beg_shadow_addr, size);
    return;
  }

  // If label is 0, releases the pages within the shadow address range, and sets
  // the shadow addresses not on the pages to be 0.
  const void *end_addr = (void *)((uptr)addr + size);
  const uptr end_shadow_addr = (uptr)__dfsan::shadow_for(end_addr);
  const uptr page_size = GetPageSizeCached();
  const uptr beg_aligned = RoundUpTo(beg_shadow_addr, page_size);
  const uptr end_aligned = RoundDownTo(end_shadow_addr, page_size);

  // dfsan_set_label can be called from the following cases
  // 1) mapped ranges by new/delete and malloc/free. This case has shadow memory
  // size > 100k, and happens less frequently.
  // 2) zero-filling internal data structures by utility libraries. This case
  // has shadow memory size < 32k, and happens more often.
  // Set kNumPagesThreshold to be 8 to avoid releasing small pages.
  const int kNumPagesThreshold = 8;
  if (beg_aligned + kNumPagesThreshold * page_size >= end_aligned)
    return WriteShadowIfDifferent(label, beg_shadow_addr, size);

  WriteShadowIfDifferent(label, beg_shadow_addr, beg_aligned - beg_shadow_addr);
  ReleaseMemoryPagesToOS(beg_aligned, end_aligned);
  WriteShadowIfDifferent(label, end_aligned, end_shadow_addr - end_aligned);
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_set_label(dfsan_label label, void *addr, uptr size) {
  __dfsan_set_label(label, addr, size);
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_add_label(dfsan_label label, void *addr, uptr size) {
  for (dfsan_label *labelp = shadow_for(addr); size != 0; --size, ++labelp)
    if (*labelp != label)
      *labelp = __dfsan_union(*labelp, label);
}

// Unlike the other dfsan interface functions the behavior of this function
// depends on the label of one of its arguments.  Hence it is implemented as a
// custom function.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
__dfsw_dfsan_get_label(long data, dfsan_label data_label,
                       dfsan_label *ret_label) {
  *ret_label = 0;
  return data_label;
}

SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
dfsan_read_label(const void *addr, uptr size) {
  if (size == 0)
    return 0;
  return __dfsan_union_load(shadow_for(addr), size);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
const struct dfsan_label_info *dfsan_get_label_info(dfsan_label label) {
  return &__dfsan_label_info[label];
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE int
dfsan_has_label(dfsan_label label, dfsan_label elem) {
  if (label == elem)
    return true;
  const dfsan_label_info *info = dfsan_get_label_info(label);
  if (info->l1 != 0) {
    return dfsan_has_label(info->l1, elem) || dfsan_has_label(info->l2, elem);
  } else {
    return false;
  }
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
dfsan_has_label_with_desc(dfsan_label label, const char *desc) {
  const dfsan_label_info *info = dfsan_get_label_info(label);
  if (info->l1 != 0) {
    return dfsan_has_label_with_desc(info->l1, desc) ||
           dfsan_has_label_with_desc(info->l2, desc);
  } else {
    return internal_strcmp(desc, info->desc) == 0;
  }
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE uptr
dfsan_get_label_count(void) {
  dfsan_label max_label_allocated =
      atomic_load(&__dfsan_last_label, memory_order_relaxed);

  return static_cast<uptr>(max_label_allocated);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
dfsan_dump_labels(int fd) {
  dfsan_label last_label =
      atomic_load(&__dfsan_last_label, memory_order_relaxed);
  for (uptr l = 1; l <= last_label; ++l) {
    char buf[64];
    internal_snprintf(buf, sizeof(buf), "%u %u %u ", l,
                      __dfsan_label_info[l].l1, __dfsan_label_info[l].l2);
    WriteToFile(fd, buf, internal_strlen(buf));
    if (__dfsan_label_info[l].l1 == 0 && __dfsan_label_info[l].desc) {
      WriteToFile(fd, __dfsan_label_info[l].desc,
                  internal_strlen(__dfsan_label_info[l].desc));
    }
    WriteToFile(fd, "\n", 1);
  }
}

#define GET_FATAL_STACK_TRACE_PC_BP(pc, bp) \
  BufferedStackTrace stack;                 \
  stack.Unwind(pc, bp, nullptr, common_flags()->fast_unwind_on_fatal);

void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void *context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  using namespace __dfsan;
  DFsanThread *t = GetCurrentThread();
  if (!t || !StackTrace::WillUseFastUnwind(request_fast)) {
    return Unwind(max_depth, pc, bp, context, 0, 0, false);
  }
  Unwind(max_depth, pc, bp, nullptr, t->stack_top(), t->stack_bottom(), true);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_print_stack_trace() {
  GET_FATAL_STACK_TRACE_PC_BP(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME());
  stack.Print();
}

void Flags::SetDefaults() {
#define DFSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "dfsan_flags.inc"
#undef DFSAN_FLAG
}

static void RegisterDfsanFlags(FlagParser *parser, Flags *f) {
#define DFSAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "dfsan_flags.inc"
#undef DFSAN_FLAG
}

static void InitializeFlags() {
  SetCommonFlagsDefaults();
  flags().SetDefaults();

  FlagParser parser;
  RegisterCommonFlags(&parser);
  RegisterDfsanFlags(&parser, &flags());
  parser.ParseStringFromEnv("DFSAN_OPTIONS");
  InitializeCommonFlags();
  if (Verbosity()) ReportUnrecognizedFlags();
  if (common_flags()->help) parser.PrintFlagDescriptions();
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_clear_arg_tls(uptr offset, uptr size) {
  internal_memset((void *)((uptr)__dfsan_arg_tls + offset), 0, size);
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_clear_thread_local_state() {
  internal_memset(__dfsan_arg_tls, 0, sizeof(__dfsan_arg_tls));
  internal_memset(__dfsan_retval_tls, 0, sizeof(__dfsan_retval_tls));
}

static void InitializePlatformEarly() {
  AvoidCVE_2016_2143();
#ifdef DFSAN_RUNTIME_VMA
  __dfsan::vmaSize =
    (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1);
  if (__dfsan::vmaSize == 39 || __dfsan::vmaSize == 42 ||
      __dfsan::vmaSize == 48) {
    __dfsan_shadow_ptr_mask = ShadowMask();
  } else {
    Printf("FATAL: DataFlowSanitizer: unsupported VMA range\n");
    Printf("FATAL: Found %d - Supported 39, 42, and 48\n", __dfsan::vmaSize);
    Die();
  }
#endif
}

static void dfsan_fini() {
  if (internal_strcmp(flags().dump_labels_at_exit, "") != 0) {
    fd_t fd = OpenFile(flags().dump_labels_at_exit, WrOnly);
    if (fd == kInvalidFd) {
      Report("WARNING: DataFlowSanitizer: unable to open output file %s\n",
             flags().dump_labels_at_exit);
      return;
    }

    Report("INFO: DataFlowSanitizer: dumping labels to %s\n",
           flags().dump_labels_at_exit);
    dfsan_dump_labels(fd);
    CloseFile(fd);
  }
}

extern "C" void dfsan_flush() {
  if (!MmapFixedSuperNoReserve(ShadowAddr(), UnusedAddr() - ShadowAddr()))
    Die();
}

static void dfsan_init(int argc, char **argv, char **envp) {
  InitializeFlags();

  ::InitializePlatformEarly();

  dfsan_flush();
  if (common_flags()->use_madv_dontdump)
    DontDumpShadowMemory(ShadowAddr(), UnusedAddr() - ShadowAddr());

  // Protect the region of memory we don't use, to preserve the one-to-one
  // mapping from application to shadow memory. But if ASLR is disabled, Linux
  // will load our executable in the middle of our unused region. This mostly
  // works so long as the program doesn't use too much memory. We support this
  // case by disabling memory protection when ASLR is disabled.
  uptr init_addr = (uptr)&dfsan_init;
  if (!(init_addr >= UnusedAddr() && init_addr < AppAddr()))
    MmapFixedNoAccess(UnusedAddr(), AppAddr() - UnusedAddr());

  InitializeInterceptors();

  // Register the fini callback to run when the program terminates successfully
  // or it is killed by the runtime.
  Atexit(dfsan_fini);
  AddDieCallback(dfsan_fini);

  // Set up threads
  DFsanTSDInit(DFsanTSDDtor);
  DFsanThread *main_thread = DFsanThread::Create(nullptr, nullptr, nullptr);
  SetCurrentThread(main_thread);
  main_thread->ThreadStart();

  __dfsan_label_info[kInitializingLabel].desc = "<init label>";
}

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used))
static void (*dfsan_init_ptr)(int, char **, char **) = dfsan_init;
#endif
