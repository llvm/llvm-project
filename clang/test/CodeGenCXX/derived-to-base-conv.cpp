// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | FileCheck %s

struct A {
  A(const A&);
  A();
  ~A();
}; 

struct B : public A {
  B();
  B(const B& Other);
  ~B();
};

struct C : public B {
  C();
  C(const C& Other);
  ~C();
}; 

struct X {
  operator B&();
  operator C&();
  X(const X&);
  X();
  ~X();
  B b;
  C c;
};

void test0_helper(A);
void test0(X x) {
  test0_helper(x);
  // CHECK-LABEL:    define{{.*}} void @_Z5test01X(
  // CHECK:      [[TMP:%.*]] = alloca [[A:%.*]], align
  // CHECK-NEXT: [[T0:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[B:%.*]]* @_ZN1XcvR1BEv(
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[B]]* [[T0]] to [[A]]*
  // CHECK-NEXT: call void @_ZN1AC1ERKS_([[A]]* {{[^,]*}} [[TMP]], [[A]]* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T1]])
  // CHECK-NEXT: call void @_Z12test0_helper1A([[A]]* noundef [[TMP]])
  // CHECK-NEXT: call void @_ZN1AD1Ev([[A]]* {{[^,]*}} [[TMP]])
  // CHECK-NEXT: ret void
}

struct Base;

struct Root {
  operator Base&();
};

struct Derived;

struct Base : Root {
  Base(const Base &);
  Base();
  operator Derived &();
};

struct Derived : Base {
};

void test1_helper(Base);
void test1(Derived bb) {
  // CHECK-LABEL:     define{{.*}} void @_Z5test17Derived(
  // CHECK-NOT: call {{.*}} @_ZN4BasecvR7DerivedEv(
  // CHECK:     call void @_ZN4BaseC1ERKS_(
  // CHECK-NOT: call {{.*}} @_ZN4BasecvR7DerivedEv(
  // CHECK:     call void @_Z12test1_helper4Base(
  test1_helper(bb);
}

// Don't crash after devirtualizing a derived-to-base conversion
// to an empty base allocated at offset zero.
// rdar://problem/11993704
class Test2a {};
class Test2b final : public virtual Test2a {};
void test2(Test2b &x) {
  Test2a &y = x;
  // CHECK-LABEL:    define{{.*}} void @_Z5test2R6Test2b(
  // CHECK:      [[X:%.*]] = alloca [[B:%.*]]*, align 8
  // CHECK-NEXT: [[Y:%.*]] = alloca [[A:%.*]]*, align 8
  // CHECK-NEXT: store [[B]]* {{%.*}}, [[B]]** [[X]], align 8
  // CHECK-NEXT: [[T0:%.*]] = load [[B]]*, [[B]]** [[X]], align 8
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[B]]* [[T0]] to [[A]]*
  // CHECK-NEXT: store [[A]]* [[T1]], [[A]]** [[Y]], align 8
  // CHECK-NEXT: ret void
}
