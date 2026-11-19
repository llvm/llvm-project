//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKGEN %s
//RUN: %if cir-enabled %{ %clang_cc1 %s -fclangir -emit-llvm -o - -triple=i686-unknown-linux -Wno-inaccessible-base | FileCheck --check-prefix=CHECKGEN %s %}
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-apple-ios6.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-apple-ios5.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKIOS5 %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=wasm32-unknown-unknown \
//RUN:   | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-unknown-fuchsia | FileCheck --check-prefix=CHECKFUCHSIA %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=aarch64-unknown-fuchsia | FileCheck --check-prefix=CHECKFUCHSIA %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i386-pc-win32 -fno-rtti | FileCheck --check-prefix=CHECKMS %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux-gnu -fctor-dtor-return-this | FileCheck --check-prefix=CHECKI686RET %s
//RUN: %if cir-enabled %{ %clang_cc1 %s -fclangir -emit-llvm -o - -triple=i686-unknown-linux-gnu -fctor-dtor-return-this -Wno-inaccessible-base | FileCheck --check-prefix=CHECKI686RET %s %}
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=aarch64-unknown-linux-gnu -fctor-dtor-return-this | FileCheck --check-prefix=CHECKAARCH64RET %s
//RUN: %if cir-enabled %{ %clang_cc1 %s -fclangir -emit-llvm -o - -triple=aarch64-unknown-linux-gnu -fctor-dtor-return-this -Wno-inaccessible-base | FileCheck --check-prefix=CHECKAARCH64RET %s %}
// FIXME: these tests crash on the bots when run with -triple=x86_64-pc-win32

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

// CHECKGEN-LABEL: define{{.*}} void @_ZN1BC2EPi(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BC1EPi(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BD2Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BD1Ev(ptr {{[^,]*}} %{{[^,)]*}})

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})

// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} %{{[^,)]*}})

// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} %{{[^,)]*}})

// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0B@@QAE@PAH@Z"(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1B@@UAE@XZ"(ptr {{[^,]*}} %{{[^,)]*}})

class C : public A, public B {
public:
  C(int *i, char *c);
  virtual ~C();
private:
  char *c_;
};

C::C(int *i, char *c) : B(i), c_(c) { }
C::~C() { }

// CHECKGEN-LABEL: define{{.*}} void @_ZN1CC2EPiPc(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CC1EPiPc(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD2Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD1Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZThn8_N1CD1Ev(ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %{{[^,)]*}})

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKARM-LABEL: define{{.*}} ptr @_ZThn8_N1CD1Ev(ptr noundef %{{[^,)]*}})
// CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZThn16_N1CD1Ev(ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKARM-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %{{[^,)]*}})
// CHECKAARCH64RET-LABEL: define{{.*}} void @_ZThn16_N1CD0Ev(ptr noundef %{{[^,)]*}})

// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZThn8_N1CD1Ev(ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %{{[^,)]*}})

// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZThn8_N1CD1Ev(ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %{{[^,)]*}})

// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZThn16_N1CD1Ev(ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} void @_ZThn16_N1CD0Ev(ptr noundef %{{[^,)]*}})

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0C@@QAE@PAHPAD@Z"(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1C@@UAE@XZ"(ptr {{[^,]*}} %{{[^,)]*}})

class D : public virtual A {
public:
  D();
  ~D();
};

D::D() { }
D::~D() { }

// CHECKGEN-LABEL: define{{.*}} void @_ZN1DC2Ev(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DC1Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DD2Ev(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DD1Ev(ptr {{[^,]*}} %{{[^,)]*}})

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})

// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} %{{[^,)]*}})

// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} %{{[^,)]*}})

// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, ptr noundef %{{[^,)]*}})
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}})

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0D@@QAE@XZ"(ptr {{[^,]*}} returned{{[^,]*}} %{{[^,)]*}}, i32 noundef %{{[^,)]*}})
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1D@@UAE@XZ"(ptr{{[^,]*}} %{{[^,)]*}})

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

// CHECKARM-LABEL,CHECKFUCHSIA-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} void @_Z15test_destructorv()

// Verify that virtual calls to destructors are not marked with a 'returned'
// this parameter at the call site...
// CHECKARM,CHECKAARCH64RET: [[VFN:%.*]] = getelementptr inbounds ptr, ptr
// CHECKARM,CHECKAARCH64RET: [[THUNK:%.*]] = load ptr, ptr [[VFN]]
// CHECKFUCHSIA: [[THUNK:%.*]] = call ptr @llvm.load.relative.i32(ptr {{.*}}, i32 0)
// CHECKARM,CHECKFUCHSIA,CHECKAARCH64RET: call noundef ptr [[THUNK]](ptr {{[^,]*}} %

// ...but static calls create declarations with 'returned' this
// CHECKARM,CHECKFUCHSIA,CHECKAARCH64RET: {{%.*}} = call noundef ptr @_ZN1ED1Ev(ptr {{[^,]*}} %

// CHECKARM,CHECKFUCHSIA,CHECKAARCH64RET: declare noundef  ptr @_ZN1ED1Ev(ptr {{[^,]*}} returned{{[^,]*}})
