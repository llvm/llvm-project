// RUN: %clang_cc1 -std=c++26 -fherbceptions -fexceptions -emit-llvm -o - %s | FileCheck %s

// A genuine two-field struct return (e.g. an iovec-style status like
// {size_t, size_t}) must not be mistaken for a throws call returning {T, i1}.
// Only a {T, i1} return carries the herbception discriminant; routing a plain
// struct through the discriminant logic would miscompile the call (and wrongly
  // fire err_herbceptions_non_throws_call_throws in a noexcept caller).

using size_t = __SIZE_TYPE__;

struct two_field_status {
  size_t position;
  size_t position_in_scatter;
};

// This is a plain two-field struct return, not a throws call: the caller
// extracts both i64 fields directly, with no herbception discriminant routing.
// CHECK: define dso_local void @_Z6calleev() #[[ATTR:[0-9]+]] {
// CHECK-NOT: herb.catch.ok
// CHECK-NOT: herb.main.trap
// CHECK: call { i64, i64 } @_Z13two_field_funv()
// CHECK: extractvalue { i64, i64 } %call, 0
// CHECK: extractvalue { i64, i64 } %call, 1
// CHECK: ret void
two_field_status two_field_fun();

void callee() noexcept {
  two_field_status s = two_field_fun();
  (void)s;
}

// CHECK: attributes #[[ATTR]] = { {{.*}} }
