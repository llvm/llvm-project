// RUN: %clang_cc1 -std=c++2d -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// A defaulted postfix increment or decrement operator copies the object,
// applies the prefix operator, and returns the copy (with NRVO).

struct Trivial {
  int v;
  Trivial &operator++();
  Trivial operator++(int) = default;
};
Trivial use_trivial(Trivial t) { return t++; }

// CHECK-LABEL: define {{.*}} @_Z11use_trivial7Trivial(
// CHECK:         call i32 @_ZN7TrivialppEi(ptr {{.*}} %t, i32 noundef 0)

// A function defaulted on its first declaration is implicitly inline.
// CHECK-LABEL: define linkonce_odr i32 @_ZN7TrivialppEi(ptr {{.*}} %this, i32 noundef %0)
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %this1, i64 4, i1 false)
// CHECK-NEXT:    call {{.*}} @_ZN7TrivialppEv(ptr {{.*}} %this1)
// CHECK:         ret i32

struct NonTrivial {
  int *p;
  NonTrivial(const NonTrivial &);
  NonTrivial(NonTrivial &&);
  ~NonTrivial();
  NonTrivial &operator--();
};
NonTrivial operator--(NonTrivial &, int) = default;
NonTrivial use_non_trivial(NonTrivial n) { return n--; }

// CHECK-LABEL: define {{.*}} @_Z15use_non_trivial10NonTrivial(
// CHECK:         call void @_ZmmR10NonTriviali(ptr {{.*}} %agg.result, ptr {{.*}} %n, i32 noundef 0)

// The copy is constructed directly in the return slot; no move constructor is
// called.
// CHECK-LABEL: define linkonce_odr void @_ZmmR10NonTriviali(ptr {{.*}} %agg.result, ptr {{.*}} %0, i32 noundef %1)
// CHECK:         call void @_ZN10NonTrivialC1ERKS_(ptr {{.*}} %agg.result, ptr {{.*}})
// CHECK:         call {{.*}} @_ZN10NonTrivialmmEv(
// CHECK-NOT:     call void @_ZN10NonTrivialC1EOS_
// CHECK:         ret void

struct Explicit {
  int v;
  Explicit &operator++();
  Explicit operator++(this Explicit &self, int) = default;
};
Explicit use_explicit(Explicit e) { return e++; }

// CHECK-LABEL: define linkonce_odr i32 @_ZNH8ExplicitppERS_i(ptr {{.*}} %self, i32 noundef %0)
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %{{.*}}, i64 4, i1 false)
// CHECK:         call {{.*}} @_ZN8ExplicitppEv(
// CHECK:         ret i32

// The requires-clause of a constrained defaulted postfix operator is part of
// its mangled name, and the implicit definition calls the more constrained
// prefix operator.
template <typename T> concept Small = sizeof(T) <= 4;
template <typename T> struct Constrained {
  T v;
  Constrained &operator++() requires Small<T>;
  Constrained &operator++();
  Constrained &operator--();
  Constrained operator++(int) requires Small<T> = default;
  friend Constrained operator--(Constrained &, int) requires Small<T> = default;
};
Constrained<int> use_constrained(Constrained<int> c) { c--; return c++; }

// CHECK-LABEL: define {{.*}} @_Z15use_constrained11ConstrainedIiE(
// CHECK:         call i32 @_ZN11ConstrainedIiEFmmERS0_iQ5SmallIT_E(ptr {{.*}} %c, i32 noundef 0)
// CHECK:         call i32 @_ZN11ConstrainedIiEppEiQ5SmallIT_E(ptr {{.*}} %c, i32 noundef 0)

// CHECK-LABEL: define linkonce_odr i32 @_ZN11ConstrainedIiEFmmERS0_iQ5SmallIT_E(ptr {{.*}} %0, i32 noundef %1)
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %{{.*}}, i64 4, i1 false)
// CHECK:         call {{.*}} @_ZN11ConstrainedIiEmmEv(
// CHECK:         ret i32

// CHECK-LABEL: define linkonce_odr i32 @_ZN11ConstrainedIiEppEiQ5SmallIT_E(ptr {{.*}} %this, i32 noundef %0)
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %this1, i64 4, i1 false)
// CHECK-NEXT:    call {{.*}} @_ZN11ConstrainedIiEppEvQ5SmallIT_E(ptr {{.*}} %this1)
// CHECK:         ret i32
