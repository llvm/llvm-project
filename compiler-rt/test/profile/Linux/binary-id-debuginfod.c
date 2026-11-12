// REQUIRES: linux, curl
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_profgen -Wl,--build-id=0x12345678 -fcoverage-mapping -O2 -shared %t/foo.c -o %t/libfoo.so
// RUN: %clang_profgen -Wl,--build-id=0xabcd1234 -fcoverage-mapping -O2 %t/main.c -L%t -lfoo -o %t/main
// RUN: env LLVM_PROFILE_FILE=%t/profdir/default_%m.profraw LD_LIBRARY_PATH=%t %run %t/main
// RUN: mkdir -p %t/buildid/12345678 %t/buildid/abcd1234
// RUN: mkdir -p %t/buildid/12345678 %t/buildid/abcd1234
// RUN: cp %t/libfoo.so %t/buildid/12345678/debuginfo
// RUN: cp %t/main %t/buildid/abcd1234/debuginfo
// RUN: llvm-profdata merge -o %t/profdata %t/profdir/default_*.profraw
// RUN: mkdir -p %t/debuginfod-cache
// RUN: env DEBUGINFOD_CACHE_PATH=%t/debuginfod-cache DEBUGINFOD_URLS=file://%t llvm-cov show -instr-profile %t/profdata | FileCheck %s
// RUN: echo "bad" > %t/libfoo.so %t/buildid/12345678/debuginfo
// RUN: echo "bad" > %t/buildid/abcd1234/debuginfo
// RUN: rm -rf %t/debuginfod-cache/*
// RUN: env DEBUGINFOD_CACHE_PTH=%t/debuginfod-cache DEBUGINFOD_URLS=file://%t llvm-cov show -instr-profile %t/profdata -debuginfod=false %t/main | FileCheck %s --check-prefix=NODEBUGINFOD

// CHECK: 1| 1|void foo(void) {}
// CHECK: 2| 1|void bar(void) {}
// CHECK: 3| 1|int main() {

// NODEBUGINFOD-NOT: foo(void) {}
// NODEBUGINFOD: main

//--- foo.c
void foo(void) {}

//--- main.c
void foo(void);
void bar(void) {}
int main() {
  foo();
  bar();
  return 0;
}
