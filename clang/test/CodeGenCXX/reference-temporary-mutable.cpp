// Tests lifetime-extended temporaries with mutable subobjects are always
// emitted as `internal global`, never `internal constant` even for:
// - `const` references with an initializer to a const expression,
// - or binding to a mutable member of an otherwise const object.
//
// Mutable members can be modified through a const object, so they can't be 
// placed in read-only memory.
//
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct WithMutable {
  int x;
  mutable int m;
  constexpr WithMutable(int X, int M) : x(X), m(M) {}
};
const WithMutable &direct_ref = WithMutable(1, 2);
void touch_direct() { direct_ref.m = 7; } 

// CHECK: @_ZGR10direct_ref_ = internal global %struct.WithMutable

struct MutableMember {
  mutable int m;
  constexpr MutableMember(int v) : m(v) {}
};
const int &member_ref = MutableMember(5).m;
void write_member() { const_cast<int &>(member_ref) = 9; }

// CHECK: @_ZGR10member_ref_ = internal global %struct.MutableMember
