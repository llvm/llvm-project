// RUN: %clang_cc1 -x c++ -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=nullability-arg | FileCheck %s -check-prefixes=ITANIUM,ALL
// RUN: %clang_cc1 -x c++ -triple x86_64-pc-windows-msvc -emit-llvm -o - %s -fsanitize=nullability-arg | FileCheck %s -check-prefixes=MSVC,ALL

namespace method_ptr {

struct S0 {
  void foo1();
};

void foo1(void (S0::*_Nonnull f)());

// ITANIUM-LABEL: @_ZN10method_ptr5test1Ev(){{.*}} {
// ITANIUM: br i1 icmp ne (i64 ptrtoint (void (%"struct.method_ptr::S0"*)* @_ZN10method_ptr2S04foo1Ev to i64), i64 0), label %[[CONT:.*]], label %[[FAIL:[^,]*]]
// ITANIUM-EMPTY:
// ITANIUM-NEXT: [[FAIL]]:
// ITANIUM-NEXT:   call void @__ubsan_handle_nullability_arg

// MSVC-LABEL: @"?test1@method_ptr@@YAXXZ"(){{.*}} {
// MSVC: br i1 true, label %[[CONT:.*]], label %[[FAIL:[^,]*]]
// MSVC-EMPTY:
// MSVC-NEXT: [[FAIL]]:
// MSVC-NEXT:   call void @__ubsan_handle_nullability_arg
void test1() {
  foo1(&S0::foo1);
}

} // namespace method_ptr

namespace data_ptr {

struct S0 {
  int field1;
};

using member_ptr = int S0::*;

void foo1(member_ptr _Nonnull);

// ITANIUM-LABEL: @_ZN8data_ptr5test1ENS_2S0E(
// MSVC-LABEL: @"?test1@data_ptr@@YAXUS0@1@@Z"(
// ALL: [[DATA_PTR_CHECK:%.*]] = icmp ne {{.*}}, -1, !nosanitize
// ALL-NEXT: br i1 [[DATA_PTR_CHECK]], label %[[CONT:.*]], label %[[FAIL:[^,]+]]
// ALL-EMPTY:
// ALL-NEXT: [[FAIL]]:
// ALL-NEXT:   call void @__ubsan_handle_nullability_arg
void test1(S0 s) {
  int S0::*member = &S0::field1;
  foo1(member);
}

} // namespace data_ptr
