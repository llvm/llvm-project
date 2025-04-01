// RUN: mkdir -p %t.d && cd %t.d
// RUN: rm -f *.profraw
// RUN: %clang_pgogen %s -o a.out

// Need to run a.out twice. On the second time, a merge will occur, which will
// trigger an mmap.
// RUN: ./a.out
// RUN: llvm-profdata show default_*.profraw --all-functions --counts --memop-sizes 2>&1 | FileCheck %s -check-prefix=PROFDATA
// RUN: env LLVM_PROFILE_NO_MMAP=1 LLVM_PROFILE_VERBOSE=1 ./a.out 2>&1 | FileCheck %s
// RUN: llvm-profdata show default_*.profraw --all-functions --counts --memop-sizes 2>&1 | FileCheck %s -check-prefix=PROFDATA2

// CHECK: could not use mmap; using fread instead
// PROFDATA: Block counts: [1]
// PROFDATA: [  0,    0,          1 ]
// PROFDATA: Maximum function count: 1
// PROFDATA2: Block counts: [2]
// PROFDATA2: [  0,    0,          2 ]
// PROFDATA2: Maximum function count: 2

int ar[8];
int main() {
  __builtin_memcpy(ar, ar + 2, ar[0]);
  __builtin_memcpy(ar, ar + 2, ar[2]);
  return ar[2];
}
