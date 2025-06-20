// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=none -mconstructor-aliases -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-DEF,EXPLICIT-DEF %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=explicit -mconstructor-aliases -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-DEF,EXPLICIT-EXP %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=all -mconstructor-aliases -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=UNSPECIFIED-EXP,EXPLICIT-EXP %s

class A {
public:
  ~A();
};

A::~A() {}

class __attribute__((visibility("default"))) B {
public:
  ~B();
};

B::~B() {}

void func() {
  A x;
  B y;
}

// A::~A() (complete object destructor)
// UNSPECIFIED-DEF:  @_ZN1AD1Ev = unnamed_addr alias
// UNSPECIFIED-EXP:  @_ZN1AD1Ev = dllexport unnamed_addr alias

// B::~B() (complete object destructor)
// EXPLICIT-DEF:  @_ZN1BD1Ev = unnamed_addr alias
// EXPLICIT-EXP:  @_ZN1BD1Ev = dllexport unnamed_addr alias

// A::~A() (base object destructor)
// UNSPECIFIED-DEF:  define void @_ZN1AD2Ev(
// UNSPECIFIED-EXP:  define dllexport void @_ZN1AD2Ev(

// B::~B() (base object destructor)
// EXPLICIT-DEF:  define void @_ZN1BD2Ev(
// EXPLICIT-EXP:  define dllexport void @_ZN1BD2Ev(
