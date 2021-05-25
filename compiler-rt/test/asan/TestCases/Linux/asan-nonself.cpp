// RUN: %clangxx_asan -O2 %s -o %t
// RUN: not %run %t g 2>&1 | FileCheck %s --check-prefix=CHECK

#include <cstdint>
using namespace std;
using uptr = unsigned long;
using u64 = uint64_t;
using u32 = uint32_t;
using s64 = int64_t;

// CHECK: AddressSanitizer: stack-buffer-overflow

// runtime interface function for nonself reporting
extern "C" void __asan_report_nonself_error(
    uptr *nonself_callstack, u32 n_nonself_callstack, uptr *nonself_addrs,
    u32 n_nonself_addrs, u64 *nonself_tids, u32 n_nonself_tids, bool is_write,
    u32 access_size, bool is_abort, const char *nonself_name,
    s64 nonself_adjust_vma, int nonself_fd, u64 nonself_file_extent_size,
    u64 nonself_file_extent_start = /*default*/ 0);

// this is a just stub function written for test coverage
void foobar() {
  int stack_arr[2];
  uptr addr[1] = {(uptr)((u64)&stack_arr[2])};
  uptr callstack[1] = {(uptr)__builtin_return_address(0)};
  u64 threads[1] = {/*dummy thread id */ 1};
  // BOOM
  __asan_report_nonself_error(callstack, 1, addr, 1, threads, 1, false,
                              4, true, "null", 0, -1, 0, 0);
  return;
}

int main() {
  foobar();
  return 0;
}
