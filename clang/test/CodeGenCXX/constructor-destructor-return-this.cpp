//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKGEN %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=thumbv7-apple-ios6.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=thumbv7-apple-ios5.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKIOS5 %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=wasm32-unknown-unknown \
//RUN:   | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=x86_64-unknown-fuchsia | FileCheck --check-prefix=CHECKFUCHSIA %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=aarch64-unknown-fuchsia | FileCheck --check-prefix=CHECKFUCHSIA %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=i386-pc-win32 -fno-rtti | FileCheck --check-prefix=CHECKMS %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=i686-unknown-linux-gnu -fctor-dtor-return-this | FileCheck --check-prefix=CHECKI686RET %s
//RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple=aarch64-unknown-linux-gnu -fctor-dtor-return-this | FileCheck --check-prefix=CHECKAARCH64RET %s
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

// CHECKGEN-LABEL: define{{.*}} void @_ZN1BC2EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BC1EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BD2Ev(%class.B* {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1BD1Ev(%class.B* {{[^,]*}} %this)

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.B* @_ZN1BC2EPi(%class.B* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.B* @_ZN1BC1EPi(%class.B* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.B* @_ZN1BD2Ev(%class.B* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.B* @_ZN1BD1Ev(%class.B* {{[^,]*}} returned{{[^,]*}} %this)

// CHECKI686RET-LABEL: define{{.*}} %class.B* @_ZN1BC2EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECKI686RET-LABEL: define{{.*}} %class.B* @_ZN1BC1EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECKI686RET-LABEL: define{{.*}} %class.B* @_ZN1BD2Ev(%class.B* {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} %class.B* @_ZN1BD1Ev(%class.B* {{[^,]*}} %this)

// CHECKIOS5-LABEL: define{{.*}} %class.B* @_ZN1BC2EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECKIOS5-LABEL: define{{.*}} %class.B* @_ZN1BC1EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECKIOS5-LABEL: define{{.*}} %class.B* @_ZN1BD2Ev(%class.B* {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} %class.B* @_ZN1BD1Ev(%class.B* {{[^,]*}} %this)

// CHECKFUCHSIA-LABEL: define{{.*}} %class.B* @_ZN1BC2EPi(%class.B* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.B* @_ZN1BC1EPi(%class.B* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.B* @_ZN1BD2Ev(%class.B* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.B* @_ZN1BD1Ev(%class.B* {{[^,]*}} returned{{[^,]*}} %this)

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef %class.B* @"??0B@@QAE@PAH@Z"(%class.B* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i)
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1B@@UAE@XZ"(%class.B* {{[^,]*}} %this)

class C : public A, public B {
public:
  C(int *i, char *c);
  virtual ~C();
private:
  char *c_;
};

C::C(int *i, char *c) : B(i), c_(c) { }
C::~C() { }

// CHECKGEN-LABEL: define{{.*}} void @_ZN1CC2EPiPc(%class.C* {{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CC1EPiPc(%class.C* {{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD2Ev(%class.C* {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD1Ev(%class.C* {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZThn8_N1CD1Ev(%class.C* noundef %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1CD0Ev(%class.C* {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(%class.C* noundef %this)

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.C* @_ZN1CC2EPiPc(%class.C* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.C* @_ZN1CC1EPiPc(%class.C* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.C* @_ZN1CD2Ev(%class.C* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.C* @_ZN1CD1Ev(%class.C* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL: define{{.*}} %class.C* @_ZThn8_N1CD1Ev(%class.C* noundef %this)
// CHECKAARCH64RET-LABEL: define{{.*}} %class.C* @_ZThn16_N1CD1Ev(%class.C* noundef %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} void @_ZN1CD0Ev(%class.C* {{[^,]*}} %this)
// CHECKARM-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(%class.C* noundef %this)
// CHECKAARCH64RET-LABEL: define{{.*}} void @_ZThn16_N1CD0Ev(%class.C* noundef %this)

// CHECKI686RET-LABEL: define{{.*}} %class.C* @_ZN1CC2EPiPc(%class.C* {{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKI686RET-LABEL: define{{.*}} %class.C* @_ZN1CC1EPiPc(%class.C* {{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKI686RET-LABEL: define{{.*}} %class.C* @_ZN1CD2Ev(%class.C* {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} %class.C* @_ZN1CD1Ev(%class.C* {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} %class.C* @_ZThn8_N1CD1Ev(%class.C* noundef %this)
// CHECKI686RET-LABEL: define{{.*}} void @_ZN1CD0Ev(%class.C* {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(%class.C* noundef %this)

// CHECKIOS5-LABEL: define{{.*}} %class.C* @_ZN1CC2EPiPc(%class.C* {{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKIOS5-LABEL: define{{.*}} %class.C* @_ZN1CC1EPiPc(%class.C* {{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKIOS5-LABEL: define{{.*}} %class.C* @_ZN1CD2Ev(%class.C* {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} %class.C* @_ZN1CD1Ev(%class.C* {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} %class.C* @_ZThn8_N1CD1Ev(%class.C* noundef %this)
// CHECKIOS5-LABEL: define{{.*}} void @_ZN1CD0Ev(%class.C* {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} void @_ZThn8_N1CD0Ev(%class.C* noundef %this)

// CHECKFUCHSIA-LABEL: define{{.*}} %class.C* @_ZN1CC2EPiPc(%class.C* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.C* @_ZN1CC1EPiPc(%class.C* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.C* @_ZN1CD2Ev(%class.C* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.C* @_ZN1CD1Ev(%class.C* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.C* @_ZThn16_N1CD1Ev(%class.C* noundef %this)
// CHECKFUCHSIA-LABEL: define{{.*}} void @_ZN1CD0Ev(%class.C* {{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} void @_ZThn16_N1CD0Ev(%class.C* noundef %this)

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef %class.C* @"??0C@@QAE@PAHPAD@Z"(%class.C* {{[^,]*}} returned{{[^,]*}} %this, i32* noundef %i, i8* noundef %c)
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1C@@UAE@XZ"(%class.C* {{[^,]*}} %this)

class D : public virtual A {
public:
  D();
  ~D();
};

D::D() { }
D::~D() { }

// CHECKGEN-LABEL: define{{.*}} void @_ZN1DC2Ev(%class.D* {{[^,]*}} %this, i8** noundef %vtt)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DC1Ev(%class.D* {{[^,]*}} %this)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DD2Ev(%class.D* {{[^,]*}} %this, i8** noundef %vtt)
// CHECKGEN-LABEL: define{{.*}} void @_ZN1DD1Ev(%class.D* {{[^,]*}} %this)

// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.D* @_ZN1DC2Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this, i8** noundef %vtt)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.D* @_ZN1DC1Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.D* @_ZN1DD2Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this, i8** noundef %vtt)
// CHECKARM-LABEL,CHECKAARCH64RET-LABEL: define{{.*}} %class.D* @_ZN1DD1Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this)

// CHECKI686RET-LABEL: define{{.*}} %class.D* @_ZN1DC2Ev(%class.D* {{[^,]*}} %this, i8** noundef %vtt)
// CHECKI686RET-LABEL: define{{.*}} %class.D* @_ZN1DC1Ev(%class.D* {{[^,]*}} %this)
// CHECKI686RET-LABEL: define{{.*}} %class.D* @_ZN1DD2Ev(%class.D* {{[^,]*}} %this, i8** noundef %vtt)
// CHECKI686RET-LABEL: define{{.*}} %class.D* @_ZN1DD1Ev(%class.D* {{[^,]*}} %this)

// CHECKIOS5-LABEL: define{{.*}} %class.D* @_ZN1DC2Ev(%class.D* {{[^,]*}} %this, i8** noundef %vtt)
// CHECKIOS5-LABEL: define{{.*}} %class.D* @_ZN1DC1Ev(%class.D* {{[^,]*}} %this)
// CHECKIOS5-LABEL: define{{.*}} %class.D* @_ZN1DD2Ev(%class.D* {{[^,]*}} %this, i8** noundef %vtt)
// CHECKIOS5-LABEL: define{{.*}} %class.D* @_ZN1DD1Ev(%class.D* {{[^,]*}} %this)

// CHECKFUCHSIA-LABEL: define{{.*}} %class.D* @_ZN1DC2Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this, i8** noundef %vtt)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.D* @_ZN1DC1Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.D* @_ZN1DD2Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this, i8** noundef %vtt)
// CHECKFUCHSIA-LABEL: define{{.*}} %class.D* @_ZN1DD1Ev(%class.D* {{[^,]*}} returned{{[^,]*}} %this)

// CHECKMS-LABEL: define dso_local x86_thiscallcc noundef %class.D* @"??0D@@QAE@XZ"(%class.D* {{[^,]*}} returned{{[^,]*}} %this, i32 noundef %is_most_derived)
// CHECKMS-LABEL: define dso_local x86_thiscallcc void @"??1D@@UAE@XZ"(%class.D* {{[^,]*}} %this)

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
// CHECKARM,CHECKAARCH64RET: [[VFN:%.*]] = getelementptr inbounds %class.E* (%class.E*)*, %class.E* (%class.E*)**
// CHECKARM,CHECKAARCH64RET: [[THUNK:%.*]] = load %class.E* (%class.E*)*, %class.E* (%class.E*)** [[VFN]]
// CHECKFUCHSIA: [[THUNK_I8:%.*]] = call i8* @llvm.load.relative.i32(i8* {{.*}}, i32 0)
// CHECKFUCHSIA: [[THUNK:%.*]] = bitcast i8* [[THUNK_I8]] to %class.E* (%class.E*)*
// CHECKARM,CHECKFUCHSIA,CHECKAARCH64RET: call noundef %class.E* [[THUNK]](%class.E* {{[^,]*}} %

// ...but static calls create declarations with 'returned' this
// CHECKARM,CHECKFUCHSIA,CHECKAARCH64RET: {{%.*}} = call noundef %class.E* @_ZN1ED1Ev(%class.E* {{[^,]*}} %

// CHECKARM,CHECKFUCHSIA,CHECKAARCH64RET: declare noundef  %class.E* @_ZN1ED1Ev(%class.E* {{[^,]*}} returned{{[^,]*}})
