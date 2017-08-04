
// RUN: %clang_cl -LD %s -Fe%t.dll -DHEAP_LIBRARY -MD
// RUN: %clang_cl %s %t.lib -Fe%t -fsanitize=address -MT
// RUN: %run %t 2>&1 | FileCheck %s

// Check that ASan does not fail when releasing allocations that occurred within
// an uninstrumented DLL.

#ifdef HEAP_LIBRARY
#include <memory>
#include <windows.h>

std::unique_ptr<int> __declspec(dllexport) myglobal(new int(42));
BOOL WINAPI DllMain(PVOID h, DWORD reason, PVOID reserved) {
  return TRUE;
}

#else

#include <memory>
extern std::unique_ptr<int> __declspec(dllimport) myglobal;
int main(int argc, char **argv) {
  printf("myglobal: %d\n", *myglobal);
  return 0;
}

#endif

// CHECK: myglobal: 42
// CHECK-NOT: ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed
