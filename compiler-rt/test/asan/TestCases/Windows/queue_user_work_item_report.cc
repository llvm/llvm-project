// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <windows.h>

HANDLE done;

DWORD CALLBACK work_item(LPVOID) {
  int subscript = -1;
  volatile char stack_buffer[42];
  stack_buffer[subscript] = 42;
// CHECK: AddressSanitizer: stack-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T1
// CHECK:   {{#0 .* work_item.*queue_user_work_item_report.cc}}:[[@LINE-3]]
// CHECK: Address [[ADDR]] is located in stack of thread T1 at offset {{.*}} in frame
// CHECK:   work_item
  SetEvent(done);
  return 0;
}

int main(int argc, char **argv) {
  done = CreateEvent(0, false, false, "job is done");
  if (!done)
    return 1;
// CHECK-NOT: Thread T1 created
  QueueUserWorkItem(&work_item, nullptr, 0);
  if (WAIT_OBJECT_0 != WaitForSingleObject(done, 10 * 1000))
    return 2;
}
