// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm %s -o - | FileCheck %s

// RUN: echo "fun:test1" > %t-func.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t-func.list -emit-llvm %s -o - | FileCheck %s --check-prefix=FUNC

// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t-file.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t-file.list -emit-llvm %s -o - | FileCheck %s --check-prefix=FILE

// RUN: echo -e "[clang]\nfun:test1\n[llvm]\nfun:test2" > %t-section.list
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t-section.list -emit-llvm %s -o - | FileCheck %s --check-prefix=SECTION

// RUN: echo -e "fun:test*\n!fun:test1" > %t-exclude.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t-exclude.list -emit-llvm %s -o - | FileCheck %s --check-prefix=EXCLUDE

// RUN: echo "!fun:test1" > %t-exclude-only.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -fprofile-list=%t-exclude-only.list -emit-llvm %s -o - | FileCheck %s --check-prefix=EXCLUDE

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
unsigned test1(void) {
  // CHECK: %pgocount = load i64, ptr @__profc_test1
  // FUNC: %pgocount = load i64, ptr @__profc_test1
  // FILE: %pgocount = load i64, ptr @__profc_test1
  // SECTION-NOT: %pgocount = load i64, ptr @__profc_test1
  // EXCLUDE-NOT: %pgocount = load i64, ptr @__profc_test1
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
unsigned test2(void) {
  // CHECK: %pgocount = load i64, ptr @__profc_test2
  // FUNC-NOT: %pgocount = load i64, ptr @__profc_test2
  // FILE: %pgocount = load i64, ptr @__profc_test2
  // SECTION: %pgocount = load i64, ptr @__profc_test2
  // EXCLUDE: %pgocount = load i64, ptr @__profc_test2
  return i - 1;
}
