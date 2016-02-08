//===-- asan_report.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file contains error reporting code.
//===----------------------------------------------------------------------===//

#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_scariness_score.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

namespace __asan {

// -------------------- User-specified callbacks ----------------- {{{1
static void (*error_report_callback)(const char*);
static char *error_message_buffer = nullptr;
static uptr error_message_buffer_pos = 0;
static BlockingMutex error_message_buf_mutex(LINKER_INITIALIZED);
static const unsigned kAsanBuggyPcPoolSize = 25;
static __sanitizer::atomic_uintptr_t AsanBuggyPcPool[kAsanBuggyPcPoolSize];

struct ReportData {
  uptr pc;
  uptr sp;
  uptr bp;
  uptr addr;
  bool is_write;
  uptr access_size;
  const char *description;
};

static bool report_happened = false;
static ReportData report_data = {};

void AppendToErrorMessageBuffer(const char *buffer) {
  BlockingMutexLock l(&error_message_buf_mutex);
  if (!error_message_buffer) {
    error_message_buffer =
      (char*)MmapOrDieQuietly(kErrorMessageBufferSize, __func__);
    error_message_buffer_pos = 0;
  }
  uptr length = internal_strlen(buffer);
  RAW_CHECK(kErrorMessageBufferSize >= error_message_buffer_pos);
  uptr remaining = kErrorMessageBufferSize - error_message_buffer_pos;
  internal_strncpy(error_message_buffer + error_message_buffer_pos,
                   buffer, remaining);
  error_message_buffer[kErrorMessageBufferSize - 1] = '\0';
  // FIXME: reallocate the buffer instead of truncating the message.
  error_message_buffer_pos += Min(remaining, length);
}

// ---------------------- Decorator ------------------------------ {{{1
class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Access()     { return Blue(); }
  const char *EndAccess()  { return Default(); }
  const char *Location()   { return Green(); }
  const char *EndLocation() { return Default(); }
  const char *Allocation()  { return Magenta(); }
  const char *EndAllocation()  { return Default(); }

  const char *ShadowByte(u8 byte) {
    switch (byte) {
      case kAsanHeapLeftRedzoneMagic:
      case kAsanHeapRightRedzoneMagic:
      case kAsanArrayCookieMagic:
        return Red();
      case kAsanHeapFreeMagic:
        return Magenta();
      case kAsanStackLeftRedzoneMagic:
      case kAsanStackMidRedzoneMagic:
      case kAsanStackRightRedzoneMagic:
      case kAsanStackPartialRedzoneMagic:
        return Red();
      case kAsanStackAfterReturnMagic:
        return Magenta();
      case kAsanInitializationOrderMagic:
        return Cyan();
      case kAsanUserPoisonedMemoryMagic:
      case kAsanContiguousContainerOOBMagic:
      case kAsanAllocaLeftMagic:
      case kAsanAllocaRightMagic:
        return Blue();
      case kAsanStackUseAfterScopeMagic:
        return Magenta();
      case kAsanGlobalRedzoneMagic:
        return Red();
      case kAsanInternalHeapMagic:
        return Yellow();
      case kAsanIntraObjectRedzone:
        return Yellow();
      default:
        return Default();
    }
  }
  const char *EndShadowByte() { return Default(); }
  const char *MemoryByte() { return Magenta(); }
  const char *EndMemoryByte() { return Default(); }
};

// ---------------------- Helper functions ----------------------- {{{1

static void PrintMemoryByte(InternalScopedString *str, const char *before,
    u8 byte, bool in_shadow, const char *after = "\n") {
  Decorator d;
  str->append("%s%s%x%x%s%s", before,
              in_shadow ? d.ShadowByte(byte) : d.MemoryByte(),
              byte >> 4, byte & 15,
              in_shadow ? d.EndShadowByte() : d.EndMemoryByte(), after);
}

static void PrintShadowByte(InternalScopedString *str, const char *before,
    u8 byte, const char *after = "\n") {
  PrintMemoryByte(str, before, byte, /*in_shadow*/true, after);
}

static void PrintShadowBytes(InternalScopedString *str, const char *before,
                             u8 *bytes, u8 *guilty, uptr n) {
  Decorator d;
  if (before) str->append("%s%p:", before, bytes);
  for (uptr i = 0; i < n; i++) {
    u8 *p = bytes + i;
    const char *before =
        p == guilty ? "[" : (p - 1 == guilty && i != 0) ? "" : " ";
    const char *after = p == guilty ? "]" : "";
    PrintShadowByte(str, before, *p, after);
  }
  str->append("\n");
}

static void PrintLegend(InternalScopedString *str) {
  str->append(
      "Shadow byte legend (one shadow byte represents %d "
      "application bytes):\n",
      (int)SHADOW_GRANULARITY);
  PrintShadowByte(str, "  Addressable:           ", 0);
  str->append("  Partially addressable: ");
  for (u8 i = 1; i < SHADOW_GRANULARITY; i++) PrintShadowByte(str, "", i, " ");
  str->append("\n");
  PrintShadowByte(str, "  Heap left redzone:       ",
                  kAsanHeapLeftRedzoneMagic);
  PrintShadowByte(str, "  Heap right redzone:      ",
                  kAsanHeapRightRedzoneMagic);
  PrintShadowByte(str, "  Freed heap region:       ", kAsanHeapFreeMagic);
  PrintShadowByte(str, "  Stack left redzone:      ",
                  kAsanStackLeftRedzoneMagic);
  PrintShadowByte(str, "  Stack mid redzone:       ",
                  kAsanStackMidRedzoneMagic);
  PrintShadowByte(str, "  Stack right redzone:     ",
                  kAsanStackRightRedzoneMagic);
  PrintShadowByte(str, "  Stack partial redzone:   ",
                  kAsanStackPartialRedzoneMagic);
  PrintShadowByte(str, "  Stack after return:      ",
                  kAsanStackAfterReturnMagic);
  PrintShadowByte(str, "  Stack use after scope:   ",
                  kAsanStackUseAfterScopeMagic);
  PrintShadowByte(str, "  Global redzone:          ", kAsanGlobalRedzoneMagic);
  PrintShadowByte(str, "  Global init order:       ",
                  kAsanInitializationOrderMagic);
  PrintShadowByte(str, "  Poisoned by user:        ",
                  kAsanUserPoisonedMemoryMagic);
  PrintShadowByte(str, "  Container overflow:      ",
                  kAsanContiguousContainerOOBMagic);
  PrintShadowByte(str, "  Array cookie:            ",
                  kAsanArrayCookieMagic);
  PrintShadowByte(str, "  Intra object redzone:    ",
                  kAsanIntraObjectRedzone);
  PrintShadowByte(str, "  ASan internal:           ", kAsanInternalHeapMagic);
  PrintShadowByte(str, "  Left alloca redzone:     ", kAsanAllocaLeftMagic);
  PrintShadowByte(str, "  Right alloca redzone:    ", kAsanAllocaRightMagic);
}

void MaybeDumpInstructionBytes(uptr pc) {
  if (!flags()->dump_instruction_bytes || (pc < GetPageSizeCached()))
    return;
  InternalScopedString str(1024);
  str.append("First 16 instruction bytes at pc: ");
  if (IsAccessibleMemoryRange(pc, 16)) {
    for (int i = 0; i < 16; ++i) {
      PrintMemoryByte(&str, "", ((u8 *)pc)[i], /*in_shadow*/false, " ");
    }
    str.append("\n");
  } else {
    str.append("unaccessible\n");
  }
  Report("%s", str.data());
}

static void PrintShadowMemoryForAddress(uptr addr) {
  if (!AddrIsInMem(addr)) return;
  uptr shadow_addr = MemToShadow(addr);
  const uptr n_bytes_per_row = 16;
  uptr aligned_shadow = shadow_addr & ~(n_bytes_per_row - 1);
  InternalScopedString str(4096 * 8);
  str.append("Shadow bytes around the buggy address:\n");
  for (int i = -5; i <= 5; i++) {
    const char *prefix = (i == 0) ? "=>" : "  ";
    PrintShadowBytes(&str, prefix, (u8 *)(aligned_shadow + i * n_bytes_per_row),
                     (u8 *)shadow_addr, n_bytes_per_row);
  }
  if (flags()->print_legend) PrintLegend(&str);
  Printf("%s", str.data());
}

static void PrintZoneForPointer(uptr ptr, uptr zone_ptr,
                                const char *zone_name) {
  if (zone_ptr) {
    if (zone_name) {
      Printf("malloc_zone_from_ptr(%p) = %p, which is %s\n",
                 ptr, zone_ptr, zone_name);
    } else {
      Printf("malloc_zone_from_ptr(%p) = %p, which doesn't have a name\n",
                 ptr, zone_ptr);
    }
  } else {
    Printf("malloc_zone_from_ptr(%p) = 0\n", ptr);
  }
}

static void DescribeThread(AsanThread *t) {
  if (t)
    DescribeThread(t->context());
}

// ---------------------- Address Descriptions ------------------- {{{1

static bool IsASCII(unsigned char c) {
  return /*0x00 <= c &&*/ c <= 0x7F;
}

static const char *MaybeDemangleGlobalName(const char *name) {
  // We can spoil names of globals with C linkage, so use an heuristic
  // approach to check if the name should be demangled.
  bool should_demangle = false;
  if (name[0] == '_' && name[1] == 'Z')
    should_demangle = true;
  else if (SANITIZER_WINDOWS && name[0] == '\01' && name[1] == '?')
    should_demangle = true;

  return should_demangle ? Symbolizer::GetOrInit()->Demangle(name) : name;
}

// Check if the global is a zero-terminated ASCII string. If so, print it.
static void PrintGlobalNameIfASCII(InternalScopedString *str,
                                   const __asan_global &g) {
  for (uptr p = g.beg; p < g.beg + g.size - 1; p++) {
    unsigned char c = *(unsigned char*)p;
    if (c == '\0' || !IsASCII(c)) return;
  }
  if (*(char*)(g.beg + g.size - 1) != '\0') return;
  str->append("  '%s' is ascii string '%s'\n", MaybeDemangleGlobalName(g.name),
              (char *)g.beg);
}

static const char *GlobalFilename(const __asan_global &g) {
  const char *res = g.module_name;
  // Prefer the filename from source location, if is available.
  if (g.location)
    res = g.location->filename;
  CHECK(res);
  return res;
}

static void PrintGlobalLocation(InternalScopedString *str,
                                const __asan_global &g) {
  str->append("%s", GlobalFilename(g));
  if (!g.location)
    return;
  if (g.location->line_no)
    str->append(":%d", g.location->line_no);
  if (g.location->column_no)
    str->append(":%d", g.location->column_no);
}

static void DescribeAddressRelativeToGlobal(uptr addr, uptr size,
                                            const __asan_global &g) {
  InternalScopedString str(4096);
  Decorator d;
  str.append("%s", d.Location());
  if (addr < g.beg) {
    str.append("%p is located %zd bytes to the left", (void *)addr,
               g.beg - addr);
  } else if (addr + size > g.beg + g.size) {
    if (addr < g.beg + g.size)
      addr = g.beg + g.size;
    str.append("%p is located %zd bytes to the right", (void *)addr,
               addr - (g.beg + g.size));
  } else {
    // Can it happen?
    str.append("%p is located %zd bytes inside", (void *)addr, addr - g.beg);
  }
  str.append(" of global variable '%s' defined in '",
             MaybeDemangleGlobalName(g.name));
  PrintGlobalLocation(&str, g);
  str.append("' (0x%zx) of size %zu\n", g.beg, g.size);
  str.append("%s", d.EndLocation());
  PrintGlobalNameIfASCII(&str, g);
  Printf("%s", str.data());
}

static bool DescribeAddressIfGlobal(uptr addr, uptr size,
                                    const char *bug_type) {
  // Assume address is close to at most four globals.
  const int kMaxGlobalsInReport = 4;
  __asan_global globals[kMaxGlobalsInReport];
  u32 reg_sites[kMaxGlobalsInReport];
  int globals_num =
      GetGlobalsForAddress(addr, globals, reg_sites, ARRAY_SIZE(globals));
  if (globals_num == 0)
    return false;
  for (int i = 0; i < globals_num; i++) {
    DescribeAddressRelativeToGlobal(addr, size, globals[i]);
    if (0 == internal_strcmp(bug_type, "initialization-order-fiasco") &&
        reg_sites[i]) {
      Printf("  registered at:\n");
      StackDepotGet(reg_sites[i]).Print();
    }
  }
  return true;
}

bool DescribeAddressIfShadow(uptr addr, AddressDescription *descr, bool print) {
  if (AddrIsInMem(addr))
    return false;
  const char *area_type = nullptr;
  if (AddrIsInShadowGap(addr)) area_type = "shadow gap";
  else if (AddrIsInHighShadow(addr)) area_type = "high shadow";
  else if (AddrIsInLowShadow(addr)) area_type = "low shadow";
  if (area_type != nullptr) {
    if (print) {
      Printf("Address %p is located in the %s area.\n", addr, area_type);
    } else {
      CHECK(descr);
      descr->region_kind = area_type;
    }
    return true;
  }
  CHECK(0 && "Address is not in memory and not in shadow?");
  return false;
}

// Return " (thread_name) " or an empty string if the name is empty.
const char *ThreadNameWithParenthesis(AsanThreadContext *t, char buff[],
                                      uptr buff_len) {
  const char *name = t->name;
  if (name[0] == '\0') return "";
  buff[0] = 0;
  internal_strncat(buff, " (", 3);
  internal_strncat(buff, name, buff_len - 4);
  internal_strncat(buff, ")", 2);
  return buff;
}

const char *ThreadNameWithParenthesis(u32 tid, char buff[],
                                      uptr buff_len) {
  if (tid == kInvalidTid) return "";
  asanThreadRegistry().CheckLocked();
  AsanThreadContext *t = GetThreadContextByTidLocked(tid);
  return ThreadNameWithParenthesis(t, buff, buff_len);
}

static void PrintAccessAndVarIntersection(const StackVarDescr &var, uptr addr,
                                          uptr access_size, uptr prev_var_end,
                                          uptr next_var_beg) {
  uptr var_end = var.beg + var.size;
  uptr addr_end = addr + access_size;
  const char *pos_descr = nullptr;
  // If the variable [var.beg, var_end) is the nearest variable to the
  // current memory access, indicate it in the log.
  if (addr >= var.beg) {
    if (addr_end <= var_end)
      pos_descr = "is inside";  // May happen if this is a use-after-return.
    else if (addr < var_end)
      pos_descr = "partially overflows";
    else if (addr_end <= next_var_beg &&
             next_var_beg - addr_end >= addr - var_end)
      pos_descr = "overflows";
  } else {
    if (addr_end > var.beg)
      pos_descr = "partially underflows";
    else if (addr >= prev_var_end &&
             addr - prev_var_end >= var.beg - addr_end)
      pos_descr = "underflows";
  }
  InternalScopedString str(1024);
  str.append("    [%zd, %zd)", var.beg, var_end);
  // Render variable name.
  str.append(" '");
  for (uptr i = 0; i < var.name_len; ++i) {
    str.append("%c", var.name_pos[i]);
  }
  str.append("'");
  if (pos_descr) {
    Decorator d;
    // FIXME: we may want to also print the size of the access here,
    // but in case of accesses generated by memset it may be confusing.
    str.append("%s <== Memory access at offset %zd %s this variable%s\n",
               d.Location(), addr, pos_descr, d.EndLocation());
  } else {
    str.append("\n");
  }
  Printf("%s", str.data());
}

bool ParseFrameDescription(const char *frame_descr,
                           InternalMmapVector<StackVarDescr> *vars) {
  CHECK(frame_descr);
  char *p;
  // This string is created by the compiler and has the following form:
  // "n alloc_1 alloc_2 ... alloc_n"
  // where alloc_i looks like "offset size len ObjectName".
  uptr n_objects = (uptr)internal_simple_strtoll(frame_descr, &p, 10);
  if (n_objects == 0)
    return false;

  for (uptr i = 0; i < n_objects; i++) {
    uptr beg  = (uptr)internal_simple_strtoll(p, &p, 10);
    uptr size = (uptr)internal_simple_strtoll(p, &p, 10);
    uptr len  = (uptr)internal_simple_strtoll(p, &p, 10);
    if (beg == 0 || size == 0 || *p != ' ') {
      return false;
    }
    p++;
    StackVarDescr var = {beg, size, p, len};
    vars->push_back(var);
    p += len;
  }

  return true;
}

bool DescribeAddressIfStack(uptr addr, uptr access_size) {
  AsanThread *t = FindThreadByStackAddress(addr);
  if (!t) return false;

  Decorator d;
  char tname[128];
  Printf("%s", d.Location());
  Printf("Address %p is located in stack of thread T%d%s", addr, t->tid(),
         ThreadNameWithParenthesis(t->tid(), tname, sizeof(tname)));

  // Try to fetch precise stack frame for this access.
  AsanThread::StackFrameAccess access;
  if (!t->GetStackFrameAccessByAddr(addr, &access)) {
    Printf("%s\n", d.EndLocation());
    return true;
  }
  Printf(" at offset %zu in frame%s\n", access.offset, d.EndLocation());

  // Now we print the frame where the alloca has happened.
  // We print this frame as a stack trace with one element.
  // The symbolizer may print more than one frame if inlining was involved.
  // The frame numbers may be different than those in the stack trace printed
  // previously. That's unfortunate, but I have no better solution,
  // especially given that the alloca may be from entirely different place
  // (e.g. use-after-scope, or different thread's stack).
#if defined(__powerpc64__) && defined(__BIG_ENDIAN__)
  // On PowerPC64 ELFv1, the address of a function actually points to a
  // three-doubleword data structure with the first field containing
  // the address of the function's code.
  access.frame_pc = *reinterpret_cast<uptr *>(access.frame_pc);
#endif
  access.frame_pc += 16;
  Printf("%s", d.EndLocation());
  StackTrace alloca_stack(&access.frame_pc, 1);
  alloca_stack.Print();

  InternalMmapVector<StackVarDescr> vars(16);
  if (!ParseFrameDescription(access.frame_descr, &vars)) {
    Printf("AddressSanitizer can't parse the stack frame "
           "descriptor: |%s|\n", access.frame_descr);
    // 'addr' is a stack address, so return true even if we can't parse frame
    return true;
  }
  uptr n_objects = vars.size();
  // Report the number of stack objects.
  Printf("  This frame has %zu object(s):\n", n_objects);

  // Report all objects in this frame.
  for (uptr i = 0; i < n_objects; i++) {
    uptr prev_var_end = i ? vars[i - 1].beg + vars[i - 1].size : 0;
    uptr next_var_beg = i + 1 < n_objects ? vars[i + 1].beg : ~(0UL);
    PrintAccessAndVarIntersection(vars[i], access.offset, access_size,
                                  prev_var_end, next_var_beg);
  }
  Printf("HINT: this may be a false positive if your program uses "
         "some custom stack unwind mechanism or swapcontext\n");
  if (SANITIZER_WINDOWS)
    Printf("      (longjmp, SEH and C++ exceptions *are* supported)\n");
  else
    Printf("      (longjmp and C++ exceptions *are* supported)\n");

  DescribeThread(t);
  return true;
}

static void DescribeAccessToHeapChunk(AsanChunkView chunk, uptr addr,
                                      uptr access_size) {
  sptr offset;
  Decorator d;
  InternalScopedString str(4096);
  str.append("%s", d.Location());
  if (chunk.AddrIsAtLeft(addr, access_size, &offset)) {
    str.append("%p is located %zd bytes to the left of", (void *)addr, offset);
  } else if (chunk.AddrIsAtRight(addr, access_size, &offset)) {
    if (offset < 0) {
      addr -= offset;
      offset = 0;
    }
    str.append("%p is located %zd bytes to the right of", (void *)addr, offset);
  } else if (chunk.AddrIsInside(addr, access_size, &offset)) {
    str.append("%p is located %zd bytes inside of", (void*)addr, offset);
  } else {
    str.append("%p is located somewhere around (this is AddressSanitizer bug!)",
               (void *)addr);
  }
  str.append(" %zu-byte region [%p,%p)\n", chunk.UsedSize(),
             (void *)(chunk.Beg()), (void *)(chunk.End()));
  str.append("%s", d.EndLocation());
  Printf("%s", str.data());
}

void DescribeHeapAddress(uptr addr, uptr access_size) {
  AsanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid()) {
    Printf("AddressSanitizer can not describe address in more detail "
           "(wild memory access suspected).\n");
    return;
  }
  DescribeAccessToHeapChunk(chunk, addr, access_size);
  CHECK(chunk.AllocTid() != kInvalidTid);
  asanThreadRegistry().CheckLocked();
  AsanThreadContext *alloc_thread =
      GetThreadContextByTidLocked(chunk.AllocTid());
  StackTrace alloc_stack = chunk.GetAllocStack();
  char tname[128];
  Decorator d;
  AsanThreadContext *free_thread = nullptr;
  if (chunk.FreeTid() != kInvalidTid) {
    free_thread = GetThreadContextByTidLocked(chunk.FreeTid());
    Printf("%sfreed by thread T%d%s here:%s\n", d.Allocation(),
           free_thread->tid,
           ThreadNameWithParenthesis(free_thread, tname, sizeof(tname)),
           d.EndAllocation());
    StackTrace free_stack = chunk.GetFreeStack();
    free_stack.Print();
    Printf("%spreviously allocated by thread T%d%s here:%s\n",
           d.Allocation(), alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
  } else {
    Printf("%sallocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
  }
  alloc_stack.Print();
  DescribeThread(GetCurrentThread());
  if (free_thread)
    DescribeThread(free_thread);
  DescribeThread(alloc_thread);
}

static void DescribeAddress(uptr addr, uptr access_size, const char *bug_type) {
  // Check if this is shadow or shadow gap.
  if (DescribeAddressIfShadow(addr))
    return;
  CHECK(AddrIsInMem(addr));
  if (DescribeAddressIfGlobal(addr, access_size, bug_type))
    return;
  if (DescribeAddressIfStack(addr, access_size))
    return;
  // Assume it is a heap address.
  DescribeHeapAddress(addr, access_size);
}

// ------------------- Thread description -------------------- {{{1

void DescribeThread(AsanThreadContext *context) {
  CHECK(context);
  asanThreadRegistry().CheckLocked();
  // No need to announce the main thread.
  if (context->tid == 0 || context->announced) {
    return;
  }
  context->announced = true;
  char tname[128];
  InternalScopedString str(1024);
  str.append("Thread T%d%s", context->tid,
             ThreadNameWithParenthesis(context->tid, tname, sizeof(tname)));
  if (context->parent_tid == kInvalidTid) {
    str.append(" created by unknown thread\n");
    Printf("%s", str.data());
    return;
  }
  str.append(
      " created by T%d%s here:\n", context->parent_tid,
      ThreadNameWithParenthesis(context->parent_tid, tname, sizeof(tname)));
  Printf("%s", str.data());
  StackDepotGet(context->stack_id).Print();
  // Recursively described parent thread if needed.
  if (flags()->print_full_thread_history) {
    AsanThreadContext *parent_context =
        GetThreadContextByTidLocked(context->parent_tid);
    DescribeThread(parent_context);
  }
}

// -------------------- Different kinds of reports ----------------- {{{1

// Use ScopedInErrorReport to run common actions just before and
// immediately after printing error report.
class ScopedInErrorReport {
 public:
  explicit ScopedInErrorReport(ReportData *report = nullptr,
                               bool fatal = false) {
    halt_on_error_ = fatal || flags()->halt_on_error;

    if (lock_.TryLock()) {
      StartReporting(report);
      return;
    }

    // ASan found two bugs in different threads simultaneously.

    u32 current_tid = GetCurrentTidOrInvalid();
    if (reporting_thread_tid_ == current_tid ||
        reporting_thread_tid_ == kInvalidTid) {
      // This is either asynch signal or nested error during error reporting.
      // Fail simple to avoid deadlocks in Report().

      // Can't use Report() here because of potential deadlocks
      // in nested signal handlers.
      const char msg[] = "AddressSanitizer: nested bug in the same thread, "
                         "aborting.\n";
      WriteToFile(kStderrFd, msg, sizeof(msg));

      internal__exit(common_flags()->exitcode);
    }

    if (halt_on_error_) {
      // Do not print more than one report, otherwise they will mix up.
      // Error reporting functions shouldn't return at this situation, as
      // they are effectively no-returns.

      Report("AddressSanitizer: while reporting a bug found another one. "
             "Ignoring.\n");

      // Sleep long enough to make sure that the thread which started
      // to print an error report will finish doing it.
      SleepForSeconds(Max(100, flags()->sleep_before_dying + 1));

      // If we're still not dead for some reason, use raw _exit() instead of
      // Die() to bypass any additional checks.
      internal__exit(common_flags()->exitcode);
    } else {
      // The other thread will eventually finish reporting
      // so it's safe to wait
      lock_.Lock();
    }

    StartReporting(report);
  }

  ~ScopedInErrorReport() {
    // Make sure the current thread is announced.
    DescribeThread(GetCurrentThread());
    // We may want to grab this lock again when printing stats.
    asanThreadRegistry().Unlock();
    // Print memory stats.
    if (flags()->print_stats)
      __asan_print_accumulated_stats();

    if (common_flags()->print_cmdline)
      PrintCmdline();

    // Copy the message buffer so that we could start logging without holding a
    // lock that gets aquired during printing.
    InternalScopedBuffer<char> buffer_copy(kErrorMessageBufferSize);
    {
      BlockingMutexLock l(&error_message_buf_mutex);
      internal_memcpy(buffer_copy.data(),
                      error_message_buffer, kErrorMessageBufferSize);
    }

    LogFullErrorReport(buffer_copy.data());

    if (error_report_callback) {
      error_report_callback(buffer_copy.data());
    }
    CommonSanitizerReportMutex.Unlock();
    reporting_thread_tid_ = kInvalidTid;
    lock_.Unlock();
    if (halt_on_error_) {
      Report("ABORTING\n");
      Die();
    }
  }

 private:
  void StartReporting(ReportData *report) {
    if (report) report_data = *report;
    report_happened = true;
    ASAN_ON_ERROR();
    // Make sure the registry and sanitizer report mutexes are locked while
    // we're printing an error report.
    // We can lock them only here to avoid self-deadlock in case of
    // recursive reports.
    asanThreadRegistry().Lock();
    CommonSanitizerReportMutex.Lock();
    reporting_thread_tid_ = GetCurrentTidOrInvalid();
    Printf("===================================================="
           "=============\n");
  }

  static StaticSpinMutex lock_;
  static u32 reporting_thread_tid_;
  bool halt_on_error_;
};

StaticSpinMutex ScopedInErrorReport::lock_;
u32 ScopedInErrorReport::reporting_thread_tid_;

void ReportStackOverflow(const SignalContext &sig) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report(
      "ERROR: AddressSanitizer: stack-overflow on address %p"
      " (pc %p bp %p sp %p T%d)\n",
      (void *)sig.addr, (void *)sig.pc, (void *)sig.bp, (void *)sig.sp,
      GetCurrentTidOrInvalid());
  Printf("%s", d.EndWarning());
  ScarinessScore::PrintSimple(15, "stack-overflow");
  GET_STACK_TRACE_SIGNAL(sig);
  stack.Print();
  ReportErrorSummary("stack-overflow", &stack);
}

void ReportDeadlySignal(const char *description, const SignalContext &sig) {
  ScopedInErrorReport in_report(/*report*/nullptr, /*fatal*/true);
  Decorator d;
  Printf("%s", d.Warning());
  Report(
      "ERROR: AddressSanitizer: %s on unknown address %p"
      " (pc %p bp %p sp %p T%d)\n",
      description, (void *)sig.addr, (void *)sig.pc, (void *)sig.bp,
      (void *)sig.sp, GetCurrentTidOrInvalid());
  Printf("%s", d.EndWarning());
  ScarinessScore SS;
  if (sig.pc < GetPageSizeCached())
    Report("Hint: pc points to the zero page.\n");
  if (sig.is_memory_access) {
    Report("The signal is caused by a %s memory access.\n",
           sig.is_write ? "WRITE" : "READ");
    if (sig.addr < GetPageSizeCached()) {
      Report("Hint: address points to the zero page.\n");
      SS.Scare(10, "null-deref");
    } else if (sig.addr == sig.pc) {
      SS.Scare(60, "wild-jump");
    } else if (sig.is_write) {
      SS.Scare(30, "wild-addr-write");
    } else {
      SS.Scare(20, "wild-addr-read");
    }
  } else {
    SS.Scare(10, "signal");
  }
  SS.Print();
  GET_STACK_TRACE_SIGNAL(sig);
  stack.Print();
  MaybeDumpInstructionBytes(sig.pc);
  Printf("AddressSanitizer can not provide additional info.\n");
  ReportErrorSummary(description, &stack);
}

void ReportDoubleFree(uptr addr, BufferedStackTrace *free_stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  u32 curr_tid = GetCurrentTidOrInvalid();
  Report("ERROR: AddressSanitizer: attempting double-free on %p in "
         "thread T%d%s:\n",
         addr, curr_tid,
         ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)));
  Printf("%s", d.EndWarning());
  CHECK_GT(free_stack->size, 0);
  ScarinessScore::PrintSimple(42, "double-free");
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("double-free", &stack);
}

void ReportNewDeleteSizeMismatch(uptr addr, uptr delete_size,
                                 BufferedStackTrace *free_stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  u32 curr_tid = GetCurrentTidOrInvalid();
  Report("ERROR: AddressSanitizer: new-delete-type-mismatch on %p in "
         "thread T%d%s:\n",
         addr, curr_tid,
         ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)));
  Printf("%s  object passed to delete has wrong type:\n", d.EndWarning());
  Printf("  size of the allocated type:   %zd bytes;\n"
         "  size of the deallocated type: %zd bytes.\n",
         asan_mz_size(reinterpret_cast<void*>(addr)), delete_size);
  CHECK_GT(free_stack->size, 0);
  ScarinessScore::PrintSimple(10, "new-delete-type-mismatch");
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("new-delete-type-mismatch", &stack);
  Report("HINT: if you don't care about these errors you may set "
         "ASAN_OPTIONS=new_delete_type_mismatch=0\n");
}

void ReportFreeNotMalloced(uptr addr, BufferedStackTrace *free_stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  char tname[128];
  u32 curr_tid = GetCurrentTidOrInvalid();
  Report("ERROR: AddressSanitizer: attempting free on address "
             "which was not malloc()-ed: %p in thread T%d%s\n", addr,
         curr_tid, ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)));
  Printf("%s", d.EndWarning());
  CHECK_GT(free_stack->size, 0);
  ScarinessScore::PrintSimple(10, "bad-free");
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("bad-free", &stack);
}

void ReportAllocTypeMismatch(uptr addr, BufferedStackTrace *free_stack,
                             AllocType alloc_type,
                             AllocType dealloc_type) {
  static const char *alloc_names[] =
    {"INVALID", "malloc", "operator new", "operator new []"};
  static const char *dealloc_names[] =
    {"INVALID", "free", "operator delete", "operator delete []"};
  CHECK_NE(alloc_type, dealloc_type);
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: alloc-dealloc-mismatch (%s vs %s) on %p\n",
        alloc_names[alloc_type], dealloc_names[dealloc_type], addr);
  Printf("%s", d.EndWarning());
  CHECK_GT(free_stack->size, 0);
  ScarinessScore::PrintSimple(10, "alloc-dealloc-mismatch");
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("alloc-dealloc-mismatch", &stack);
  Report("HINT: if you don't care about these errors you may set "
         "ASAN_OPTIONS=alloc_dealloc_mismatch=0\n");
}

void ReportMallocUsableSizeNotOwned(uptr addr, BufferedStackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: attempting to call "
             "malloc_usable_size() for pointer which is "
             "not owned: %p\n", addr);
  Printf("%s", d.EndWarning());
  stack->Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("bad-malloc_usable_size", stack);
}

void ReportSanitizerGetAllocatedSizeNotOwned(uptr addr,
                                             BufferedStackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: attempting to call "
             "__sanitizer_get_allocated_size() for pointer which is "
             "not owned: %p\n", addr);
  Printf("%s", d.EndWarning());
  stack->Print();
  DescribeHeapAddress(addr, 1);
  ReportErrorSummary("bad-__sanitizer_get_allocated_size", stack);
}

void ReportStringFunctionMemoryRangesOverlap(const char *function,
                                             const char *offset1, uptr length1,
                                             const char *offset2, uptr length2,
                                             BufferedStackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  char bug_type[100];
  internal_snprintf(bug_type, sizeof(bug_type), "%s-param-overlap", function);
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: %s: "
             "memory ranges [%p,%p) and [%p, %p) overlap\n", \
             bug_type, offset1, offset1 + length1, offset2, offset2 + length2);
  Printf("%s", d.EndWarning());
  ScarinessScore::PrintSimple(10, bug_type);
  stack->Print();
  DescribeAddress((uptr)offset1, length1, bug_type);
  DescribeAddress((uptr)offset2, length2, bug_type);
  ReportErrorSummary(bug_type, stack);
}

void ReportStringFunctionSizeOverflow(uptr offset, uptr size,
                                      BufferedStackTrace *stack) {
  ScopedInErrorReport in_report;
  Decorator d;
  const char *bug_type = "negative-size-param";
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: %s: (size=%zd)\n", bug_type, size);
  Printf("%s", d.EndWarning());
  ScarinessScore::PrintSimple(10, bug_type);
  stack->Print();
  DescribeAddress(offset, size, bug_type);
  ReportErrorSummary(bug_type, stack);
}

void ReportBadParamsToAnnotateContiguousContainer(uptr beg, uptr end,
                                                  uptr old_mid, uptr new_mid,
                                                  BufferedStackTrace *stack) {
  ScopedInErrorReport in_report;
  Report("ERROR: AddressSanitizer: bad parameters to "
         "__sanitizer_annotate_contiguous_container:\n"
         "      beg     : %p\n"
         "      end     : %p\n"
         "      old_mid : %p\n"
         "      new_mid : %p\n",
         beg, end, old_mid, new_mid);
  uptr granularity = SHADOW_GRANULARITY;
  if (!IsAligned(beg, granularity))
    Report("ERROR: beg is not aligned by %d\n", granularity);
  stack->Print();
  ReportErrorSummary("bad-__sanitizer_annotate_contiguous_container", stack);
}

void ReportODRViolation(const __asan_global *g1, u32 stack_id1,
                        const __asan_global *g2, u32 stack_id2) {
  ScopedInErrorReport in_report;
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: odr-violation (%p):\n", g1->beg);
  Printf("%s", d.EndWarning());
  InternalScopedString g1_loc(256), g2_loc(256);
  PrintGlobalLocation(&g1_loc, *g1);
  PrintGlobalLocation(&g2_loc, *g2);
  Printf("  [1] size=%zd '%s' %s\n", g1->size,
         MaybeDemangleGlobalName(g1->name), g1_loc.data());
  Printf("  [2] size=%zd '%s' %s\n", g2->size,
         MaybeDemangleGlobalName(g2->name), g2_loc.data());
  if (stack_id1 && stack_id2) {
    Printf("These globals were registered at these points:\n");
    Printf("  [1]:\n");
    StackDepotGet(stack_id1).Print();
    Printf("  [2]:\n");
    StackDepotGet(stack_id2).Print();
  }
  Report("HINT: if you don't care about these errors you may set "
         "ASAN_OPTIONS=detect_odr_violation=0\n");
  InternalScopedString error_msg(256);
  error_msg.append("odr-violation: global '%s' at %s",
                   MaybeDemangleGlobalName(g1->name), g1_loc.data());
  ReportErrorSummary(error_msg.data());
}

// ----------------------- CheckForInvalidPointerPair ----------- {{{1
static NOINLINE void
ReportInvalidPointerPair(uptr pc, uptr bp, uptr sp, uptr a1, uptr a2) {
  ScopedInErrorReport in_report;
  const char *bug_type = "invalid-pointer-pair";
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: invalid-pointer-pair: %p %p\n", a1, a2);
  Printf("%s", d.EndWarning());
  GET_STACK_TRACE_FATAL(pc, bp);
  stack.Print();
  DescribeAddress(a1, 1, bug_type);
  DescribeAddress(a2, 1, bug_type);
  ReportErrorSummary(bug_type, &stack);
}

static INLINE void CheckForInvalidPointerPair(void *p1, void *p2) {
  if (!flags()->detect_invalid_pointer_pairs) return;
  uptr a1 = reinterpret_cast<uptr>(p1);
  uptr a2 = reinterpret_cast<uptr>(p2);
  AsanChunkView chunk1 = FindHeapChunkByAddress(a1);
  AsanChunkView chunk2 = FindHeapChunkByAddress(a2);
  bool valid1 = chunk1.IsValid();
  bool valid2 = chunk2.IsValid();
  if ((valid1 != valid2) || (valid1 && valid2 && !chunk1.Eq(chunk2))) {
    GET_CALLER_PC_BP_SP;                                              \
    return ReportInvalidPointerPair(pc, bp, sp, a1, a2);
  }
}
// ----------------------- Mac-specific reports ----------------- {{{1

void ReportMacMzReallocUnknown(uptr addr, uptr zone_ptr, const char *zone_name,
                               BufferedStackTrace *stack) {
  ScopedInErrorReport in_report;
  Printf("mz_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n",
             addr);
  PrintZoneForPointer(addr, zone_ptr, zone_name);
  stack->Print();
  DescribeHeapAddress(addr, 1);
}

// -------------- SuppressErrorReport -------------- {{{1
// Avoid error reports duplicating for ASan recover mode.
static bool SuppressErrorReport(uptr pc) {
  if (!common_flags()->suppress_equal_pcs) return false;
  for (unsigned i = 0; i < kAsanBuggyPcPoolSize; i++) {
    uptr cmp = atomic_load_relaxed(&AsanBuggyPcPool[i]);
    if (cmp == 0 && atomic_compare_exchange_strong(&AsanBuggyPcPool[i], &cmp,
                                                   pc, memory_order_relaxed))
      return false;
    if (cmp == pc) return true;
  }
  Die();
}

static void PrintContainerOverflowHint() {
  Printf("HINT: if you don't care about these errors you may set "
         "ASAN_OPTIONS=detect_container_overflow=0.\n"
         "If you suspect a false positive see also: "
         "https://github.com/google/sanitizers/wiki/"
         "AddressSanitizerContainerOverflow.\n");
}

void ReportGenericError(uptr pc, uptr bp, uptr sp, uptr addr, bool is_write,
                        uptr access_size, u32 exp, bool fatal) {
  if (!fatal && SuppressErrorReport(pc)) return;
  ENABLE_FRAME_POINTER;
  ScarinessScore SS;

  if (access_size) {
    if (access_size <= 9) {
      char desr[] = "?-byte";
      desr[0] = '0' + access_size;
      SS.Scare(access_size + access_size / 2, desr);
    } else if (access_size >= 10) {
      SS.Scare(15, "multi-byte");
    }
    is_write ? SS.Scare(20, "write") : SS.Scare(1, "read");
  }

  // Optimization experiments.
  // The experiments can be used to evaluate potential optimizations that remove
  // instrumentation (assess false negatives). Instead of completely removing
  // some instrumentation, compiler can emit special calls into runtime
  // (e.g. __asan_report_exp_load1 instead of __asan_report_load1) and pass
  // mask of experiments (exp).
  // The reaction to a non-zero value of exp is to be defined.
  (void)exp;

  // Determine the error type.
  const char *bug_descr = "unknown-crash";
  u8 shadow_val = 0;
  if (AddrIsInMem(addr)) {
    u8 *shadow_addr = (u8*)MemToShadow(addr);
    // If we are accessing 16 bytes, look at the second shadow byte.
    if (*shadow_addr == 0 && access_size > SHADOW_GRANULARITY)
      shadow_addr++;
    // If we are in the partial right redzone, look at the next shadow byte.
    if (*shadow_addr > 0 && *shadow_addr < 128)
      shadow_addr++;
    bool far_from_bounds = false;
    shadow_val = *shadow_addr;
    int bug_type_score = 0;
    switch (shadow_val) {
      case kAsanHeapLeftRedzoneMagic:
      case kAsanHeapRightRedzoneMagic:
      case kAsanArrayCookieMagic:
        bug_descr = "heap-buffer-overflow";
        bug_type_score = 10;
        far_from_bounds = shadow_addr[-1] > 127 && shadow_addr[1] > 127;
        break;
      case kAsanHeapFreeMagic:
        bug_descr = "heap-use-after-free";
        bug_type_score = 20;
        break;
      case kAsanStackLeftRedzoneMagic:
        bug_descr = "stack-buffer-underflow";
        bug_type_score = 25;
        far_from_bounds = shadow_addr[-1] > 127 && shadow_addr[1] > 127;
        break;
      case kAsanInitializationOrderMagic:
        bug_descr = "initialization-order-fiasco";
        bug_type_score = 1;
        break;
      case kAsanStackMidRedzoneMagic:
      case kAsanStackRightRedzoneMagic:
      case kAsanStackPartialRedzoneMagic:
        bug_descr = "stack-buffer-overflow";
        bug_type_score = 25;
        far_from_bounds = shadow_addr[-1] > 127 && shadow_addr[1] > 127;
        break;
      case kAsanStackAfterReturnMagic:
        bug_descr = "stack-use-after-return";
        bug_type_score = 30;
        break;
      case kAsanUserPoisonedMemoryMagic:
        bug_descr = "use-after-poison";
        bug_type_score = 10;
        break;
      case kAsanContiguousContainerOOBMagic:
        bug_descr = "container-overflow";
        bug_type_score = 10;
        break;
      case kAsanStackUseAfterScopeMagic:
        bug_descr = "stack-use-after-scope";
        bug_type_score = 10;
        break;
      case kAsanGlobalRedzoneMagic:
        bug_descr = "global-buffer-overflow";
        bug_type_score = 10;
        far_from_bounds = shadow_addr[-1] > 127 && shadow_addr[1] > 127;
        break;
      case kAsanIntraObjectRedzone:
        bug_descr = "intra-object-overflow";
        bug_type_score = 10;
        break;
      case kAsanAllocaLeftMagic:
      case kAsanAllocaRightMagic:
        bug_descr = "dynamic-stack-buffer-overflow";
        bug_type_score = 25;
        far_from_bounds = shadow_addr[-1] > 127 && shadow_addr[1] > 127;
        break;
    }
    SS.Scare(bug_type_score, bug_descr);
    if (far_from_bounds)
      SS.Scare(10, "far-from-bounds");
  }

  ReportData report = { pc, sp, bp, addr, (bool)is_write, access_size,
                        bug_descr };
  ScopedInErrorReport in_report(&report, fatal);

  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: AddressSanitizer: %s on address "
             "%p at pc %p bp %p sp %p\n",
             bug_descr, (void*)addr, pc, bp, sp);
  Printf("%s", d.EndWarning());

  u32 curr_tid = GetCurrentTidOrInvalid();
  char tname[128];
  Printf("%s%s of size %zu at %p thread T%d%s%s\n",
         d.Access(),
         access_size ? (is_write ? "WRITE" : "READ") : "ACCESS",
         access_size, (void*)addr, curr_tid,
         ThreadNameWithParenthesis(curr_tid, tname, sizeof(tname)),
         d.EndAccess());

  SS.Print();
  GET_STACK_TRACE_FATAL(pc, bp);
  stack.Print();

  DescribeAddress(addr, access_size, bug_descr);
  if (shadow_val == kAsanContiguousContainerOOBMagic)
    PrintContainerOverflowHint();
  ReportErrorSummary(bug_descr, &stack);
  PrintShadowMemoryForAddress(addr);
}

}  // namespace __asan

// --------------------------- Interface --------------------- {{{1
using namespace __asan;  // NOLINT

void __asan_report_error(uptr pc, uptr bp, uptr sp, uptr addr, int is_write,
                         uptr access_size, u32 exp) {
  ENABLE_FRAME_POINTER;
  bool fatal = flags()->halt_on_error;
  ReportGenericError(pc, bp, sp, addr, is_write, access_size, exp, fatal);
}

void NOINLINE __asan_set_error_report_callback(void (*callback)(const char*)) {
  BlockingMutexLock l(&error_message_buf_mutex);
  error_report_callback = callback;
}

void __asan_describe_address(uptr addr) {
  // Thread registry must be locked while we're describing an address.
  asanThreadRegistry().Lock();
  DescribeAddress(addr, 1, "");
  asanThreadRegistry().Unlock();
}

int __asan_report_present() {
  return report_happened ? 1 : 0;
}

uptr __asan_get_report_pc() {
  return report_data.pc;
}

uptr __asan_get_report_bp() {
  return report_data.bp;
}

uptr __asan_get_report_sp() {
  return report_data.sp;
}

uptr __asan_get_report_address() {
  return report_data.addr;
}

int __asan_get_report_access_type() {
  return report_data.is_write ? 1 : 0;
}

uptr __asan_get_report_access_size() {
  return report_data.access_size;
}

const char *__asan_get_report_description() {
  return report_data.description;
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_ptr_sub(void *a, void *b) {
  CheckForInvalidPointerPair(a, b);
}
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_ptr_cmp(void *a, void *b) {
  CheckForInvalidPointerPair(a, b);
}
} // extern "C"

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
// Provide default implementation of __asan_on_error that does nothing
// and may be overriden by user.
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE NOINLINE
void __asan_on_error() {}
#endif
