// Make sure we can throw exceptions from work items executed via
// QueueUserWorkItem.
//
// RUN: %clangxx_asan %s -o %t.exe
// RUN: %run %t.exe 2>&1 | FileCheck %s

#include <windows.h>
#include <stdio.h>

void ThrowAndCatch();

__declspec(noinline)
void Throw() {
  fprintf(stderr, "Throw\n");
// CHECK: Throw
  throw 1;
}

void ThrowAndCatch() {
  int local;
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch\n");
// CHECK: Catch
  }
}

HANDLE done;

DWORD CALLBACK work_item(LPVOID) {
  ThrowAndCatch();
  SetEvent(done);
  return 0;
}

int main(int argc, char **argv) {
  done = CreateEvent(0, false, false, "job is done");
  if (!done)
    return 1;
  QueueUserWorkItem(&work_item, nullptr, 0);
  unsigned wait_result = WaitForSingleObject(done, 10 * 1000);
  if (wait_result == WAIT_ABANDONED)
    fprintf(stderr, "Timed out\n");
  if (wait_result != WAIT_OBJECT_0) {
    fprintf(stderr, "Wait for work item failed\n");
    return 2;
  }
  fprintf(stderr, "Done!\n");
// CHECK: Done!
}
