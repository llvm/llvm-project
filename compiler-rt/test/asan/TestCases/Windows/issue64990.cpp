// Repro for the issue #64990: Asan with Windows EH generates __asan_xxx runtime calls without required funclet tokens
// RUN: %clang_cl_asan %Od %s -EHsc %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

char buff1[6] = "hello";
char buff2[6] = "hello";

int main(int argc, char **argv) {
  try {
    throw 1;
  } catch (...) {
    // Make asan generate call to __asan_memcpy inside the EH pad.
    __builtin_memcpy(buff1, buff2 + 3, 6);
  }
  return 0;
}

// CHECK: SUMMARY: AddressSanitizer: global-buffer-overflow {{.*}} in __asan_memcpy
