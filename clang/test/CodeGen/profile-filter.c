// RUN: %clang_cc1 -fprofile-instrument=clang -emit-llvm %s -o - | FileCheck %s

// RUN: echo "fun:test1" > %t-func.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fprofile-list=%t-func.list -emit-llvm %s -o - | FileCheck %s --check-prefix=FUNC

// RUN: echo -e "src:%s" | sed -e 's/\\/\\\\/g' > %t-file.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fprofile-list=%t-file.list -emit-llvm %s -o - | FileCheck %s --check-prefix=FILE

// RUN: echo -e "[clang]\nfun:test1\n[llvm]\nfun:test2" > %t-section.list
// RUN: %clang_cc1 -fprofile-instrument=llvm -fprofile-list=%t-section.list -emit-llvm %s -o - | FileCheck %s --check-prefix=SECTION

// RUN: echo -e "fun:test*\n!fun:test1" | sed -e 's/\\/\\\\/g' > %t-exclude.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fprofile-list=%t-exclude.list -emit-llvm %s -o - | FileCheck %s --check-prefix=EXCLUDE

// RUN: echo -e "!fun:test1" | sed -e 's/\\/\\\\/g' > %t-exclude-only.list
// RUN: %clang_cc1 -fprofile-instrument=clang -fprofile-list=%t-exclude-only.list -emit-llvm %s -o - | FileCheck %s --check-prefix=EXCLUDE

unsigned i;

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
unsigned test1() {
  // CHECK: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test1, i64 0, i64 0), align 8
  // FUNC: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test1, i64 0, i64 0), align 8
  // FILE: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test1, i64 0, i64 0), align 8
  // SECTION-NOT: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test1, i64 0, i64 0), align 8
  // EXCLUDE-NOT: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test1, i64 0, i64 0), align 8
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
unsigned test2() {
  // CHECK: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test2, i64 0, i64 0), align 8
  // FUNC-NOT: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test2, i64 0, i64 0), align 8
  // FILE: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test2, i64 0, i64 0), align 8
  // SECTION: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test2, i64 0, i64 0), align 8
  // EXCLUDE: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_test2, i64 0, i64 0), align 8
  return i - 1;
}
