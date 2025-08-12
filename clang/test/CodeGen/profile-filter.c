// RUN: rm -rf %t && split-file %s %t

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm %t/main.c -o - | FileCheck %s
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t/func.list -emit-llvm %t/main.c -o - | FileCheck %s --check-prefix=FUNC

// RUN: echo "src:%t/main.c" | sed -e 's/\\/\\\\/g' > %t-file.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t-file.list -emit-llvm %t/main.c -o - | FileCheck %s --check-prefix=FILE
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t/section.list -emit-llvm %t/main.c -o - | FileCheck %s --check-prefix=SECTION
// RUN: %clang_cc1 -fprofile-instrument=sample-coldcov -fprofile-list=%t/cold-func.list -emit-llvm %t/main.c -o - | FileCheck %s --check-prefix=COLDCOV 
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t/exclude.list -emit-llvm %t/main.c -o - | FileCheck %s --check-prefix=EXCLUDE
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t/exclude-only.list -emit-llvm %t/main.c -o - | FileCheck %s --check-prefix=EXCLUDE

//--- func.list
fun:test1

//--- section.list
[clang]
fun:test1
[llvm]
fun:test2

//--- cold-func.list
[sample-coldcov]
fun:test*
!fun:test2

//--- exclude.list
fun:test*
!fun:test1

//--- exclude-only.list
!fun:test1

//--- main.c
unsigned i;

// CHECK: test1
// CHECK: test2
// FUNC: test1
// FUNC-NOT: test2
// FILE: test1
// FILE: test2
// EXCLUDE-NOT: test1
// EXCLUDE: test2

// CHECK-NOT: noprofile
// CHECK: @test1
// FUNC-NOT: noprofile
// FUNC: @test1
// FILE-NOT: noprofile
// FILE: @test1
// SECTION: noprofile
// SECTION: @test1
// EXCLUDE: noprofile
// EXCLUDE: @test1
// COLDCOV-NOT: noprofile
// COLDCOV: @test1
unsigned test1(void) {
  // CHECK: %pgocount = load i64, ptr @__profc_{{_?}}test1
  // FUNC: %pgocount = load i64, ptr @__profc_{{_?}}test1
  // FILE: %pgocount = load i64, ptr @__profc_{{_?}}test1
  // SECTION-NOT: %pgocount = load i64, ptr @__profc_{{_?}}test1
  // EXCLUDE-NOT: %pgocount = load i64, ptr @__profc_{{_?}}test1
  return i + 1;
}

// CHECK-NOT: noprofile
// CHECK: @test2
// FUNC: noprofile
// FUNC: @test2
// FILE-NOT: noprofile
// FILE: @test2
// SECTION-NOT: noprofile
// SECTION: @test2
// EXCLUDE-NOT: noprofile
// EXCLUDE: @test2
// COLDCOV: noprofile
// COLDCOV: @test2
unsigned test2(void) {
  // CHECK: %pgocount = load i64, ptr @__profc_{{_?}}test2
  // FUNC-NOT: %pgocount = load i64, ptr @__profc_{{_?}}test2
  // FILE: %pgocount = load i64, ptr @__profc_{{_?}}test2
  // SECTION: %pgocount = load i64, ptr @__profc_{{_?}}test2
  // EXCLUDE: %pgocount = load i64, ptr @__profc_{{_?}}test2
  return i - 1;
}
