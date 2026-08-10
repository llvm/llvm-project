//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKGEN %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-apple-ios6.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-apple-ios5.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKIOS5 %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=wasm32-unknown-unknown \
//RUN:   | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-unknown-fuchsia | FileCheck --check-prefix=CHECKFUCHSIA %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=aarch64-unknown-fuchsia | FileCheck --check-prefix=CHECKFUCHSIA %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i386-pc-win32 -fno-rtti | FileCheck --check-prefix=CHECKMS %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux-gnu -fctor-dtor-return-this | FileCheck --check-prefix=CHECKI686RET %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=aarch64-unknown-linux-gnu -fctor-dtor-return-this | FileCheck --check-prefix=CHECKAARCH64RET %s
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

// CHECKGEN-LABEL: define{{.*}} void @_ZN1BC2EPi(ptr {{[^,]*}} %this, ptr noundef %i)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BC1EPi(ptr {{[^,]*}} %this, ptr noundef %i)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BD2Ev(ptr {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BD1Ev(ptr {{[^,]*}} %this)

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)

// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} %this, ptr noundef %i)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} %this, ptr noundef %i)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} %this)

// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} %this, ptr noundef %i)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} %this, ptr noundef %i)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} %this)

// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BC2EPi(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BC1EPi(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1BD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0B@@QAE@PAH@Z"(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i)
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1B@@UAE@XZ"(ptr {{[^,]*}} %this)

class C : public A, public B {
public:
  C(int *i, char *c);
  virtual ~C();
private:
  char *c_;
};

C::C(int *i, char *c) : B(i), c_(c) { }
C::~C() { }

// CHECKGEN-LABEL: define{{.*}} void @_ZN1CC2EPiPc(ptr {{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CC1EPiPc(ptr {{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD2Ev(ptr {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD1Ev(ptr {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZThn8_N1CD1Ev(ptr noundef %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %this)

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL: define{{.*}} ptr @_ZThn8_N1CD1Ev(ptr noundef %this)
// CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZThn16_N1CD1Ev(ptr noundef %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %this)
// CHECKARM-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %this)
// CHECKAARCH64RET-LABEL: define{{.*}} void @_ZThn16_N1CD0Ev(ptr noundef %this)

// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZThn8_N1CD1Ev(ptr noundef %this)
// CHECKI686RET-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %this)

// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZThn8_N1CD1Ev(ptr noundef %this)
// CHECKIOS5-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(ptr noundef %this)

// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CC2EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CC1EPiPc(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1CD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZThn16_N1CD1Ev(ptr noundef %this)
// CHECKFUCHSIA-LABEL: define{{.*}} void @_ZN1CD0Ev(ptr {{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} void @_ZThn16_N1CD0Ev(ptr noundef %this)

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0C@@QAE@PAHPAD@Z"(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %i, ptr noundef %c)
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1C@@UAE@XZ"(ptr {{[^,]*}} %this)

class D : public virtual A {
public:
  D();
  ~D();
};

D::D() { }
D::~D() { }

// CHECKGEN-LABEL: define{{.*}} void @_ZN1DC2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DC1Ev(ptr {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DD2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DD1Ev(ptr {{[^,]*}} %this)

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %vtt)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %vtt)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)

// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt)
// CHECKI686RET-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} %this)

// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt)
// CHECKIOS5-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} %this)

// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DC2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %vtt)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DC1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DD2Ev(ptr {{[^,]*}} returned{{[^,]*}} %this, ptr noundef %vtt)
// CHECKFUCHSIA-LABEL: define{{.*}} ptr @_ZN1DD1Ev(ptr {{[^,]*}} returned{{[^,]*}} %this)

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef ptr @"??0D@@QAE@XZ"(ptr {{[^,]*}} returned{{[^,]*}} %this, i32 noundef %is_most_derived)
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1D@@UAE@XZ"(ptr{{[^,]*}} %this)

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
