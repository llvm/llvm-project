// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// When a default member initializer containing a lambda is rebuilt at its point
// of use (CWG1815/CWG2631 aggregate initialization), the rebuilt lambda must
// still initialize its by-copy init-captures. TransformInitializer lowers the
// capture initializer to its syntactic form (a stripped copy source, a
// ParenListExpr for direct-init, or an InitListExpr for list-init), so the
// initialization has to be performed again. Previously it was not, leaving a
// bare initializer that crashed CodeGen for a trivially destructible closure
// and silently left the capture uninitialized otherwise.

template <typename T> int consume(const T &);

struct NonTrivialCopy {
  NonTrivialCopy();
  NonTrivialCopy(const NonTrivialCopy &);
  NonTrivialCopy &operator=(const NonTrivialCopy &);
};

// Copy-init capture with a trivially destructible closure: this is the
// configuration that previously crashed in CodeGen (EmitAggregateCopy of a
// non-trivial type).
struct CopyInitCapture {
  NonTrivialCopy member;
  int value = consume([copy = member] {});
};

CopyInitCapture kCopyInitCapture{};

// CHECK-LABEL: define internal void @__cxx_global_var_init(
// CHECK: call void @_ZN14NonTrivialCopyC1ERKS_(ptr {{.*}} %ref.tmp, ptr {{.*}} @kCopyInitCapture)

// Direct-init capture: TransformInitializer reverts this to a ParenListExpr, so
// the capture initialization must be performed again to rebuild the construct.
struct DirectInitCapture {
  NonTrivialCopy member;
  int value = consume([copy(member)] {});
};

DirectInitCapture kDirectInitCapture{};

// CHECK-LABEL: define internal void @__cxx_global_var_init.1(
// CHECK: call void @_ZN14NonTrivialCopyC1ERKS_(ptr {{.*}} %ref.tmp, ptr {{.*}} @kDirectInitCapture)

// List-init capture: TransformInitializer reverts this to an InitListExpr.
struct ListInitCapture {
  NonTrivialCopy member;
  int value = consume([copy{member}] {});
};

ListInitCapture kListInitCapture{};

// CHECK-LABEL: define internal void @__cxx_global_var_init.2(
// CHECK: call void @_ZN14NonTrivialCopyC1ERKS_(ptr {{.*}} %ref.tmp, ptr {{.*}} @kListInitCapture)

struct NonTrivialWithDtor {
  NonTrivialWithDtor();
  NonTrivialWithDtor(const NonTrivialWithDtor &);
  ~NonTrivialWithDtor();
};

// Non-trivially destructible closure: the capture must be copy-constructed and
// the closure destroyed at the end of the full-expression.
struct NonTriviallyDestructibleClosure {
  NonTrivialWithDtor member;
  int value = consume([copy = member] {});
};

NonTriviallyDestructibleClosure kNonTriviallyDestructibleClosure{};

// CHECK-LABEL: define internal void @__cxx_global_var_init.3(
// CHECK: call void @_ZN18NonTrivialWithDtorC1ERKS_(ptr {{.*}} %ref.tmp, ptr {{.*}} @kNonTriviallyDestructibleClosure)
// CHECK: call void @_ZN31NonTriviallyDestructibleClosure5valueMUlvE_D1Ev(ptr {{.*}} %ref.tmp)

// CHECK-LABEL: define linkonce_odr void @_ZN31NonTriviallyDestructibleClosure5valueMUlvE_D2Ev(
// CHECK: call void @_ZN18NonTrivialWithDtorD1Ev(ptr {{.*}} %this1)
