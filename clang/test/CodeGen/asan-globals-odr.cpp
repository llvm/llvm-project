// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefixes=CHECK,INDICATOR1,GLOB_ALIAS_INDICATOR,ALIAS1
// RUN: %clang_cc1 -fsanitize=address -fno-sanitize-address-use-odr-indicator -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefixes=CHECK,INDICATOR0,GLOB_VAR,ALIAS0
// RUN: %clang_cc1 -fsanitize=address -fno-sanitize-address-use-odr-indicator -fsanitize-address-use-odr-indicator -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefixes=CHECK,INDICATOR1,GLOB_ALIAS_INDICATOR,ALIAS1
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-use-odr-indicator -fno-sanitize-address-use-odr-indicator -emit-llvm -o - -triple x86_64-linux %s | FileCheck %s --check-prefixes=CHECK,INDICATOR0,GLOB_VAR,ALIAS0

// No alias on Windows but indicators should work.
// RUN: %clang_cc1 -fsanitize=address -fno-sanitize-address-use-odr-indicator -emit-llvm -o - -triple x86_64-windows-msvc %s | FileCheck %s --check-prefixes=CHECK,GLOB_VAR,ALIAS0
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-windows-msvc %s | FileCheck %s --check-prefixes=CHECK,INDICATOR1,GLOB_VAR_INDICATOR,ALIAS0

int global;

int main() {
  return global;
}

// CHECK: [[VAR:@.*global.*]] ={{.*}} global { i32, [28 x i8] } zeroinitializer, align 32

// INDICATOR0-NOT: __odr_asan_gen
// INDICATOR1: [[ODR:@.*__odr_asan_gen_.*global.*]] = global i8 0, align 1

// GLOB_VAR: @0 = internal global {{.*}} [[VAR]] to i64), {{.*}}, i64 0 }]
// GLOB_VAR_INDICATOR: @0 = internal global {{.*}} [[VAR]] to i64), {{.*}}, i64 ptrtoint (ptr [[ODR]] to i64) }]
// GLOB_ALIAS_INDICATOR: @0 = internal global {{.*}} @1 to i64), {{.*}}, i64 ptrtoint (ptr [[ODR]] to i64) }]

// ALIAS0-NOT: private alias
// ALIAS1: @1 = private alias {{.*}} [[VAR]]

// CHECK: call void @__asan_register_globals(i64 ptrtoint (ptr @0 to i64), i64 1)
// CHECK: call void @__asan_unregister_globals(i64 ptrtoint (ptr @0 to i64), i64 1)
