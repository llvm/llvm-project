// REQUIRES: linux
// RUN: split-file %s %t
// RUN: %clang_profgen -Wl,--build-id=0x12345678 -fcoverage-mapping -O2 -shared %t/foo.c -o %t/libfoo.so
// RUN: %clang_profgen -Wl,--build-id=0xabcd1234 -fcoverage-mapping -O2 %t/main.c -L%t -lfoo -o %t.main
// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw LD_LIBRARY_PATH=%t %run %t.main
// RUN: mkdir -p %t/.build-id/12 %t/.build-id/ab
// RUN: cp %t/libfoo.so %t/.build-id/12/345678.debug
// RUN: cp %t.main %t/.build-id/ab/cd1234.debug
// RUN: llvm-profdata merge -o %t.profdata %t.profdir/default_*.profraw
// RUN: llvm-cov show -instr-profile %t.profdata -debug-file-directory %t | FileCheck %s
// RUN: llvm-cov show -instr-profile %t.profdata %t/libfoo.so -sources %t/foo.c -object %t.main | FileCheck %s --check-prefix=FOO-ONLY
// RUN: llvm-cov show -instr-profile %t.profdata -debug-file-directory %t -sources %t/foo.c | FileCheck %s --check-prefix=FOO-ONLY
// RUN: llvm-cov show -instr-profile %t.profdata -debug-file-directory %t %t/libfoo.so -sources %t/foo.c | FileCheck %s --check-prefix=FOO-ONLY
// RUN: echo "bad" > %t/.build-id/ab/cd1234.debug
// RUN: llvm-cov show -instr-profile %t.profdata -debug-file-directory %t %t.main | FileCheck %s
// RUN: not llvm-cov show -instr-profile %t.profdata -debug-file-directory %t/empty 2>&1 | FileCheck %s --check-prefix=NODATA

// CHECK: 1| 1|void foo(void) {}
// CHECK: 2| 1|void bar(void) {}
// CHECK: 3| 1|int main() {

// FOO-ONLY: 1| 1|void foo(void) {}
// NODATA: error: Failed to load coverage: '': No coverage data found

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
