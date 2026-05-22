// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s
//
//===----------------------------------------------------------------------===//
// Tests that CIRGen does not emit illegal copies for return values of
// non-trivially-copyable struct types.
//
// Because ABI calling convention lowering is deferred in CIR, we do not use
// sret parameters for functions that return a non-copyable struct. This is
// acceptable as long as CIRGen does not introduce a store/load round-trip
// through __retval, which would constitute an illegal copy for types whose
// copy constructor is deleted.
//
// The expected CIR pattern for such functions is a direct forwarding of the
// call result to cir.return, without any alloca/store/load indirection.
//
// Uses -std=c++17 for guaranteed copy elision (P0135), so that returning a
// prvalue of a non-copyable type is well-formed without requiring an
// accessible copy or move constructor.
//===----------------------------------------------------------------------===//

// --- Test 1: Forwarding return of a struct with deleted copy+move ctor ---
// This is the core bug from issue #198602. C++17 guaranteed copy elision
// ensures return foo() is well-formed. CIR must not introduce a store/load
// round-trip through __retval.
namespace deleted_copy_ctor {
struct S {
  S();
  S(const S &) = delete;
  S(S &&) = delete;
  ~S();
};

S foo();
S bar() { return foo(); }

// CIR-LABEL: cir.func {{.*}} @_ZN17deleted_copy_ctor3barEv
// CIR-NOT:     __retval
// CIR:         %{{[0-9]+}} = cir.call @_ZN17deleted_copy_ctor3fooEv() : () -> !rec{{.*}}
// CIR-NEXT:    cir.return %{{[0-9]+}} : !rec{{.*}}

// LLVM-LABEL: define {{.*}} @_ZN17deleted_copy_ctor3barEv(

// OGCG-LABEL: define {{.*}} void @_ZN17deleted_copy_ctor3barEv(
// OGCG:         call void @_ZN17deleted_copy_ctor3fooEv(ptr
// OGCG-NEXT:    ret void
}

// --- Test 2: Forwarding return with only copy deleted, move available ---
// When only the copy constructor is deleted but a move constructor exists,
// the return should still not go through __retval.
namespace copy_deleted {
struct S {
  S();
  S(const S &) = delete;
  S(S &&);
  ~S();
};

S foo();
S bar() { return foo(); }

// CIR-LABEL: cir.func {{.*}} @_ZN12copy_deleted3barEv
// CIR-NOT:     __retval
// CIR:         %{{[0-9]+}} = cir.call @_ZN12copy_deleted3fooEv() : () -> !rec{{.*}}
// CIR-NEXT:    cir.return %{{[0-9]+}} : !rec{{.*}}

// LLVM-LABEL: define {{.*}} @_ZN12copy_deleted3barEv(

// OGCG-LABEL: define {{.*}} void @_ZN12copy_deleted3barEv(
// OGCG:         call void @_ZN12copy_deleted3fooEv(ptr
// OGCG-NEXT:    ret void
}

// --- Test 3: Forwarding return with copy deleted by a member ---
namespace deleted_by_member {
struct B {
  B();
  B(const B &) = delete;
  B(B &&) = delete;
  ~B();
};
struct A {
  A();
  B b;
  ~A();
};

A foo();
A bar() { return foo(); }

// CIR-LABEL: cir.func {{.*}} @_ZN16deleted_by_member3barEv
// CIR-NOT:     __retval
// CIR:         %{{[0-9]+}} = cir.call @_ZN16deleted_by_member3fooEv() : () -> !rec{{.*}}
// CIR-NEXT:    cir.return %{{[0-9]+}} : !rec{{.*}}

// LLVM-LABEL: define {{.*}} @_ZN16deleted_by_member3barEv(

// OGCG-LABEL: define {{.*}} void @_ZN16deleted_by_member3barEv(
// OGCG:         call void @_ZN16deleted_by_member3fooEv(ptr
// OGCG-NEXT:    ret void
}

// --- Test 4: Forwarding return with copy deleted by a base class ---
namespace deleted_by_base {
struct B {
  B();
  B(const B &) = delete;
  B(B &&) = delete;
  ~B();
};
struct A : B {
  A();
  ~A();
};

A foo();
A bar() { return foo(); }

// CIR-LABEL: cir.func {{.*}} @_ZN14deleted_by_base3barEv
// CIR-NOT:     __retval
// CIR:         %{{[0-9]+}} = cir.call @_ZN14deleted_by_base3fooEv() : () -> !rec{{.*}}
// CIR-NEXT:    cir.return %{{[0-9]+}} : !rec{{.*}}

// LLVM-LABEL: define {{.*}} @_ZN14deleted_by_base3barEv(

// OGCG-LABEL: define {{.*}} void @_ZN14deleted_by_base3barEv(
// OGCG:         call void @_ZN14deleted_by_base3fooEv(ptr
// OGCG-NEXT:    ret void
}

// --- Test 5: Implicitly deleted copy ctor (struct with reference member) ---
// A class with a reference member has an implicitly deleted copy
// constructor and an implicitly deleted move constructor (until C++20).
// This should also avoid the __retval pattern.
namespace implicitly_deleted {
struct S {
  S();
  int &ref;
  ~S();
};

S foo();
S bar() { return foo(); }

// CIR-LABEL: cir.func {{.*}} @_ZN18implicitly_deleted3barEv
// CIR-NOT:     __retval
// CIR:         %{{[0-9]+}} = cir.call @_ZN18implicitly_deleted3fooEv() : () -> !rec{{.*}}
// CIR-NEXT:    cir.return %{{[0-9]+}} : !rec{{.*}}

// LLVM-LABEL: define {{.*}} @_ZN18implicitly_deleted3barEv(

// OGCG-LABEL: define {{.*}} void @_ZN18implicitly_deleted3barEv(
// OGCG:         call void @_ZN18implicitly_deleted3fooEv(ptr
// OGCG-NEXT:    ret void
}

// --- Test 6: Trivially-copyable struct (control group) ---
// For trivially-copyable types, the __retval store/load pattern is still
// acceptable since no copy constructor semantics are violated. This test
// verifies the fix does not regress existing behavior for trivial types.
namespace trivially_copyable {
struct S {
  int x;
  double y;
};

S make();
S use() { return make(); }

// CIR-LABEL: cir.func {{.*}} @_ZN18trivially_copyable3useEv
// CIR:         cir.alloca !rec{{.*}}, !cir.ptr<!rec{{.*}}, ["__retval"]
// CIR:         cir.store
// CIR:         cir.load
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN18trivially_copyable3useEv(

// OGCG-LABEL: define {{.*}} @_ZN18trivially_copyable3useEv(
}

// --- Test 7: Return of struct with trivial copy ctor but non-trivial dtor ---
// A struct with a trivial copy constructor and a non-trivial destructor is
// not trivially copyable per the standard, but its copy constructor is still
// trivial and accessible, so the __retval store/load is a valid
// memcpy-equivalent. The pattern is OK.
namespace trivial_copy_non_trivial_dtor {
struct S {
  int x;
  ~S();
};

S make();
S use() { return make(); }

// CIR-LABEL: cir.func {{.*}} @_ZN26trivial_copy_non_trivial_dtor3useEv
// CIR:         cir.alloca !rec{{.*}}, !cir.ptr<!rec{{.*}}, ["__retval"]
// CIR:         cir.store
// CIR:         cir.load
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN26trivial_copy_non_trivial_dtor3useEv(

// OGCG-LABEL: define {{.*}} @_ZN26trivial_copy_non_trivial_dtor3useEv(
}

// --- Test 8: Non-copyable struct returned via ternary expression ---
// Conditional return should also avoid the store/load.
namespace conditional_return {
struct S {
  S();
  S(const S &) = delete;
  S(S &&) = delete;
  ~S();
};

S a();
S b();
S pick(bool c) { return c ? a() : b(); }

// CIR-LABEL: cir.func {{.*}} @_ZN18conditional_return4pickEb
// CIR-NOT:     __retval
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN18conditional_return4pickEb(

// OGCG-LABEL: define {{.*}} void @_ZN18conditional_return4pickEb(
}

// --- Test 9: Trivially-copyable struct returned through ternary ---
// Even with non-trivial control flow, trivial types can use __retval.
namespace conditional_trivial {
struct S {
  int x;
};

S a();
S b();
S pick(bool c) { return c ? a() : b(); }

// CIR-LABEL: cir.func {{.*}} @_ZN19conditional_trivial4pickEb
// CIR:         cir.alloca !rec{{.*}}, !cir.ptr<!rec{{.*}}, ["__retval"]
// CIR:         cir.store
// CIR:         cir.load
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN19conditional_trivial4pickEb(

// OGCG-LABEL: define {{.*}} @_ZN19conditional_trivial4pickEb(
}

// --- Test 10: Non-copyable struct returned from multiple branches ---
// Each return site must independently avoid the __retval pattern.
namespace multiple_returns {
struct S {
  S();
  S(const S &) = delete;
  S(S &&) = delete;
  ~S();
};

S a();
S b();
S c();
S pick(int n) {
  if (n > 0) return a();
  if (n < 0) return b();
  return c();
}

// CIR-LABEL: cir.func {{.*}} @_ZN16multiple_returns4pickEi
// CIR-NOT:     __retval
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN16multiple_returns4pickEi(

// OGCG-LABEL: define {{.*}} void @_ZN16multiple_returns4pickEi(
}

// --- Test 11: Virtual function returning non-copyable struct ---
// Virtual dispatch should not introduce __retval copies.
namespace virtual_return {
struct S {
  S();
  S(const S &) = delete;
  S(S &&) = delete;
  ~S();
};

struct Base {
  virtual S foo();
};

struct Derived : Base {
  S foo() override;
};

S caller(Base &b) { return b.foo(); }

// CIR-LABEL: cir.func {{.*}} @_ZN13virtual_return6callerERNS_4BaseE
// CIR-NOT:     __retval
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN13virtual_return6callerERNS_4BaseE(

// OGCG-LABEL: define {{.*}} void @_ZN13virtual_return6callerERNS_4BaseE(
}

// --- Test 12: Function template returning non-copyable struct ---
// Template instantiation must also avoid __retval.
namespace templated_return {
struct S {
  S();
  S(const S &) = delete;
  S(S &&) = delete;
  ~S();
};

template <typename T>
T make();

S bar() { return make<S>(); }

// CIR-LABEL: cir.func {{.*}} @_ZN16templated_return3barEv
// CIR-NOT:     __retval
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} @_ZN16templated_return3barEv(

// OGCG-LABEL: define {{.*}} void @_ZN16templated_return3barEv(
}
