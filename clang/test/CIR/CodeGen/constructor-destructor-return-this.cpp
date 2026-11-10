// RUN: %clang_cc1 %s -fclangir -emit-cir -o - -triple=aarch64-unknown-linux-gnu -fctor-dtor-return-this -Wno-inaccessible-base | FileCheck --check-prefix=CIRRET %s
// RUN: %clang_cc1 %s -fclangir -emit-cir -o - -triple=i686-unknown-linux-gnu -fctor-dtor-return-this -Wno-inaccessible-base | FileCheck --check-prefix=CIRRET %s
// RUN: %clang_cc1 %s -fclangir -emit-cir -o - -triple=x86_64-unknown-linux-gnu -Wno-inaccessible-base | FileCheck --check-prefix=CIRGEN %s
// Make sure we attach the 'returned' attribute to the 'this' parameter of
// constructors and destructors which return this (and only these cases)

class A {
public:
  A();
  virtual ~A();

private:
  int x_;
};

class B : public A {
public:
  B(int *i);
  virtual ~B();

private:
  int *i_;
};

B::B(int *i) : i_(i) { }
B::~B() { }


// CIRRET-LABEL: cir.func {{.*}} @_ZN1BC2EPi(
// CIRRET-SAME:    %[[THIS_ARG:[^:]*]]: !cir.ptr<!rec_B> {{{[^}]*}}llvm.returned{{[^}]*}}}
// CIRRET-SAME:    -> (!cir.ptr<!rec_B> {llvm.noundef})
// CIRRET:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIRRET:         %[[THIS_ADDR:.*]] = cir.alloca "this"
// CIRRET:         cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIRRET:         %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIRRET-NEXT:    cir.store {{.*}}%[[THIS]], %[[RETVAL]]
// CIRRET:         cir.call @_ZN1AC2Ev({{.*}}) : ({{.*}}) -> (!cir.ptr<!rec_A> {llvm.noundef})
// CIRRET:         %[[RET:.*]] = cir.load %[[RETVAL]]
// CIRRET-NEXT:    cir.return implicit %[[RET]] : !cir.ptr<!rec_B>
// CIRRET-LABEL: cir.func {{.*}} @_ZN1BC1EPi(
// CIRRET-SAME:    llvm.returned
// CIRRET:         cir.call @_ZN1BC2EPi({{.*}}) : ({{.*}}) -> (!cir.ptr<!rec_B> {llvm.noundef})
// CIRRET-LABEL: cir.func {{.*}} @_ZN1BD2Ev(
// CIRRET-SAME:    llvm.returned
// CIRRET-SAME:    -> (!cir.ptr<!rec_B> {llvm.noundef})
// CIRRET:         cir.call @_ZN1AD2Ev({{.*}}) nothrow : ({{.*}}) -> (!cir.ptr<!rec_A> {llvm.noundef})
// CIRRET-LABEL: cir.func {{.*}} @_ZN1BD1Ev(
// CIRRET-SAME:    llvm.returned
// CIRRET:         cir.call @_ZN1BD2Ev({{.*}}) nothrow : ({{.*}}) -> (!cir.ptr<!rec_B> {llvm.noundef})
// The deleting destructor is called virtually, so it still returns void.
// CIRRET-LABEL: cir.func {{.*}} @_ZN1BD0Ev(
// CIRRET-NOT:     llvm.returned
// CIRRET:         cir.call @_ZN1BD1Ev({{.*}}) nothrow : ({{.*}}) -> (!cir.ptr<!rec_B> {llvm.noundef})

// Without the flag, nothing returns 'this'.
// CIRGEN-LABEL: cir.func {{.*}} @_ZN1BC2EPi(
// CIRGEN-NOT:     llvm.returned
// CIRGEN-LABEL: cir.func {{.*}} @_ZN1BC1EPi(
// CIRGEN-NOT:     llvm.returned
// CIRGEN-LABEL: cir.func {{.*}} @_ZN1BD2Ev(
// CIRGEN-NOT:     llvm.returned
// CIRGEN-LABEL: cir.func {{.*}} @_ZN1BD1Ev(
// CIRGEN-NOT:     llvm.returned

class C : public A, public B {
public:
  C(int *i, char *c);
  virtual ~C();
private:
  char *c_;
};

C::C(int *i, char *c) : B(i), c_(c) { }
C::~C() { }


// CIRRET-LABEL: cir.func {{.*}} @_ZN1CC2EPiPc(
// CIRRET-SAME:    llvm.returned
// CIRRET-SAME:    -> (!cir.ptr<!rec_C> {llvm.noundef})
// CIRRET:         cir.call @_ZN1AC2Ev({{.*}}) : ({{.*}}) -> (!cir.ptr<!rec_A> {llvm.noundef})
// CIRRET:         cir.call @_ZN1BC2EPi({{.*}}) : ({{.*}}) -> (!cir.ptr<!rec_B> {llvm.noundef})
// CIRRET-LABEL: cir.func {{.*}} @_ZN1CC1EPiPc(
// CIRRET-SAME:    llvm.returned
// CIRRET-LABEL: cir.func {{.*}} @_ZN1CD2Ev(
// CIRRET-SAME:    llvm.returned
// CIRRET-LABEL: cir.func {{.*}} @_ZN1CD1Ev(
// CIRRET-SAME:    llvm.returned
// The thunk forwards the returned 'this' value; its parameter is not 'returned'.
// CIRRET-LABEL: cir.func {{.*}} @_ZThn{{8|16}}_N1CD1Ev(
// CIRRET-NOT:     llvm.returned
// CIRRET-SAME:    -> (!cir.ptr<!rec_C> {llvm.noundef})
// CIRRET:         %[[CALL:.*]] = cir.call @_ZN1CD1Ev({{.*}}) {{.*}}-> (!cir.ptr<!rec_C> {llvm.noundef})
// CIRRET:         cir.store {{.*}}%[[CALL]]
// CIRRET-LABEL: cir.func {{.*}} @_ZN1CD0Ev(
// CIRRET-NOT:     llvm.returned
// CIRRET-LABEL: cir.func {{.*}} @_ZThn{{8|16}}_N1CD0Ev(
// CIRRET-NOT:     llvm.returned

class D : public virtual A {
public:
  D();
  ~D();
};

D::D() { }
D::~D() { }


// The VTT parameter stays after 'this'.
// CIRRET-LABEL: cir.func {{.*}} @_ZN1DC2Ev(
// CIRRET-SAME:    llvm.returned{{.*}}%{{.*}}: !cir.ptr<!cir.ptr<!void>> {llvm.noundef}
// CIRRET-SAME:    -> (!cir.ptr<!rec_D> {llvm.noundef})
// CIRRET-LABEL: cir.func {{.*}} @_ZN1DC1Ev(
// CIRRET-SAME:    llvm.returned
// CIRRET-LABEL: cir.func {{.*}} @_ZN1DD2Ev(
// CIRRET-SAME:    llvm.returned{{.*}}%{{.*}}: !cir.ptr<!cir.ptr<!void>> {llvm.noundef}
// CIRRET-LABEL: cir.func {{.*}} @_ZN1DD1Ev(
// CIRRET-SAME:    llvm.returned

class E {
public:
  E();
  virtual ~E();
};

E* gete();

void test_destructor() {
  const E& e1 = E();
  E* e2 = gete();
  e2->~E();
}

// Verify that virtual calls to destructors are not marked with a 'returned'
// this parameter at the call site...

// ...but static calls create declarations with 'returned' this

// CIRRET-LABEL: cir.func {{.*}} @_Z15test_destructorv()
// CIRRET:         cir.call @_ZN1EC1Ev({{.*}}) : ({{.*}}) -> (!cir.ptr<!rec_E> {llvm.noundef})
// CIRRET:         %[[FN:.*]] = cir.load {{.*}} : {{.*}}, !cir.ptr<!cir.func<(!cir.ptr<!rec_E>) -> !cir.ptr<!rec_E>>>
// CIRRET:         cir.call %[[FN]]({{.*}}) nothrow : ({{.*}}) -> (!cir.ptr<!rec_E> {llvm.noundef})
// CIRRET:         cir.call @_ZN1ED1Ev({{.*}}) nothrow : ({{.*}}) -> (!cir.ptr<!rec_E> {llvm.noundef})
// CIRRET:       cir.func private unnamed_addr @_ZN1EC1Ev({{.*}}llvm.returned{{.*}}) -> (!cir.ptr<!rec_E> {llvm.noundef})
// CIRRET:       cir.func private unnamed_addr @_ZN1ED1Ev({{.*}}llvm.returned{{.*}}) -> (!cir.ptr<!rec_E> {llvm.noundef})
