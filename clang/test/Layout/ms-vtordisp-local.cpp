// RUN: %clang_cc1 -fms-extensions -fexceptions -fcxx-exceptions -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>&1 | FileCheck %s

struct Base {
  virtual ~Base() {}
  virtual void BaseFunc() {}
};

#pragma vtordisp(0)

struct Container {
  static void f() try {
    #pragma vtordisp(2)
    struct HasVtorDisp : virtual Base {
      virtual ~HasVtorDisp() {}
      virtual void Func() {}
    };

    int x[sizeof(HasVtorDisp)];

    // HasVtorDisp: vtordisp because of pragma right before it.
    //
    // CHECK: *** Dumping AST Record Layout
    // CHECK: *** Dumping AST Record Layout
    // CHECK-NEXT:          0 | struct HasVtorDisp
    // CHECK-NEXT:          0 |   (HasVtorDisp vftable pointer)
    // CHECK-NEXT:          8 |   (HasVtorDisp vbtable pointer)
    // CHECK-NEXT:         20 |   (vtordisp for vbase Base)
    // CHECK-NEXT:         24 |   struct Base (virtual base)
    // CHECK-NEXT:         24 |     (Base vftable pointer)
    // CHECK-NEXT:            | [sizeof=32, align=8,
    // CHECK-NEXT:            |  nvsize=16, nvalign=8]
  } catch (...) {
  }
};

struct NoVtorDisp1 : virtual Base {
  virtual ~NoVtorDisp1() {}
  virtual void Func() {}
};

int x1[sizeof(NoVtorDisp1)];

// NoVtroDisp1: no vtordisp because of pragma disabling it.
//
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct NoVtorDisp1
// CHECK-NEXT:          0 |   (NoVtorDisp1 vftable pointer)
// CHECK-NEXT:          8 |   (NoVtorDisp1 vbtable pointer)
// CHECK-NEXT:         16 |   struct Base (virtual base)
// CHECK-NEXT:         16 |     (Base vftable pointer)
// CHECK-NEXT:            | [sizeof=24, align=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=8]

struct Container2 {
  static void f1() {
    // Local pragma #1 - must be disabled on exit from f1().
    #pragma vtordisp(push, 2)
    struct HasVtorDisp1 : virtual Base {
      virtual ~HasVtorDisp1() {}
      virtual void Func() {}
    };

    int x2[sizeof(HasVtorDisp1)];

    // HasVtorDisp1: vtordisp because of pragma right before it.
    //
    // CHECK: *** Dumping AST Record Layout
    // CHECK-NEXT:          0 | struct HasVtorDisp1
    // CHECK-NEXT:          0 |   (HasVtorDisp1 vftable pointer)
    // CHECK-NEXT:          8 |   (HasVtorDisp1 vbtable pointer)
    // CHECK-NEXT:         20 |   (vtordisp for vbase Base)
    // CHECK-NEXT:         24 |   struct Base (virtual base)
    // CHECK-NEXT:         24 |     (Base vftable pointer)
    // CHECK-NEXT:            | [sizeof=32, align=8,
    // CHECK-NEXT:            |  nvsize=16, nvalign=8]

    struct InnerContainer {
      static void g1() {
        struct HasVtorDisp2 : virtual Base {
          virtual ~HasVtorDisp2() {}
          virtual void Func() {}
        };

        int x3[sizeof(HasVtorDisp2)];

        // HasVtorDisp2: vtordisp because of vtordisp(2) in f1().
        //
        // CHECK: *** Dumping AST Record Layout
        // CHECK-NEXT:          0 | struct HasVtorDisp2
        // CHECK-NEXT:          0 |   (HasVtorDisp2 vftable pointer)
        // CHECK-NEXT:          8 |   (HasVtorDisp2 vbtable pointer)
        // CHECK-NEXT:         20 |   (vtordisp for vbase Base)
        // CHECK-NEXT:         24 |   struct Base (virtual base)
        // CHECK-NEXT:         24 |     (Base vftable pointer)
        // CHECK-NEXT:            | [sizeof=32, align=8,
        // CHECK-NEXT:            |  nvsize=16, nvalign=8]

        // Local pragma #2 - must be disabled on exit from g1().
        #pragma vtordisp(push, 0)
        struct NoVtorDisp2 : virtual Base {
          virtual ~NoVtorDisp2() {}
          virtual void Func() {}
        };

        int x4[sizeof(NoVtorDisp2)];

        // NoVtroDisp2: no vtordisp because of vtordisp(0) in g1().
        //
        // CHECK: *** Dumping AST Record Layout
        // CHECK-NEXT:          0 | struct NoVtorDisp2
        // CHECK-NEXT:          0 |   (NoVtorDisp2 vftable pointer)
        // CHECK-NEXT:          8 |   (NoVtorDisp2 vbtable pointer)
        // CHECK-NEXT:         16 |   struct Base (virtual base)
        // CHECK-NEXT:         16 |     (Base vftable pointer)
        // CHECK-NEXT:            | [sizeof=24, align=8,
        // CHECK-NEXT:            |  nvsize=16, nvalign=8]
      }

      static void g2() {
        struct HasVtorDisp3 : virtual Base {
          virtual ~HasVtorDisp3() {}
          virtual void Func() {}
        };

        int x5[sizeof(HasVtorDisp3)];

        // HasVtorDisp3: vtordisp because of vtordisp(2) in f1(),
        //               local vtordisp(0) in g1() is disabled.
        //
        // CHECK: *** Dumping AST Record Layout
        // CHECK-NEXT:          0 | struct HasVtorDisp3
        // CHECK-NEXT:          0 |   (HasVtorDisp3 vftable pointer)
        // CHECK-NEXT:          8 |   (HasVtorDisp3 vbtable pointer)
        // CHECK-NEXT:         20 |   (vtordisp for vbase Base)
        // CHECK-NEXT:         24 |   struct Base (virtual base)
        // CHECK-NEXT:         24 |     (Base vftable pointer)
        // CHECK-NEXT:            | [sizeof=32, align=8,
        // CHECK-NEXT:            |  nvsize=16, nvalign=8]
      }
    };

    struct HasVtorDisp4 : virtual Base {
      virtual ~HasVtorDisp4() {}
      virtual void Func() {}
    };

    int x6[sizeof(HasVtorDisp4)];

    // HasVtorDisp4: vtordisp because of vtordisp(2) in f1(),
    //               local vtordisp(0) in g1() is disabled,
    //               g2() has no pragmas - stack is not affected.
    //
    // CHECK: *** Dumping AST Record Layout
    // CHECK-NEXT:          0 | struct HasVtorDisp4
    // CHECK-NEXT:          0 |   (HasVtorDisp4 vftable pointer)
    // CHECK-NEXT:          8 |   (HasVtorDisp4 vbtable pointer)
    // CHECK-NEXT:         20 |   (vtordisp for vbase Base)
    // CHECK-NEXT:         24 |   struct Base (virtual base)
    // CHECK-NEXT:         24 |     (Base vftable pointer)
    // CHECK-NEXT:            | [sizeof=32, align=8,
    // CHECK-NEXT:            |  nvsize=16, nvalign=8]

    InnerContainer::g1();
    InnerContainer::g2();
  }

  static void f2() {
    struct NoVtorDisp3 : virtual Base {
      virtual ~NoVtorDisp3() {}
      virtual void Func() {}
    };

    int x7[sizeof(NoVtorDisp3)];

    // NoVtroDisp3: no vtordisp because of global pragma (0),
    //              local vtordisp(2) is disabled on exit from f1().
    //
    // CHECK: *** Dumping AST Record Layout
    // CHECK-NEXT:          0 | struct NoVtorDisp3
    // CHECK-NEXT:          0 |   (NoVtorDisp3 vftable pointer)
    // CHECK-NEXT:          8 |   (NoVtorDisp3 vbtable pointer)
    // CHECK-NEXT:         16 |   struct Base (virtual base)
    // CHECK-NEXT:         16 |     (Base vftable pointer)
    // CHECK-NEXT:            | [sizeof=24, align=8,
    // CHECK-NEXT:            |  nvsize=16, nvalign=8]
  }
};

struct Container3 {
  #pragma vtordisp(2)
  struct HasVtorDisp5 : virtual Base {
    virtual ~HasVtorDisp5() {}
    virtual void Func() {}
  };

  int x8[sizeof(HasVtorDisp5)];

  // HasVtorDisp5: vtordisp because of pragma right before it.
  //
  // CHECK: *** Dumping AST Record Layout
  // CHECK-NEXT:          0 | struct Container3::HasVtorDisp5
  // CHECK-NEXT:          0 |   (HasVtorDisp5 vftable pointer)
  // CHECK-NEXT:          8 |   (HasVtorDisp5 vbtable pointer)
  // CHECK-NEXT:         20 |   (vtordisp for vbase Base)
  // CHECK-NEXT:         24 |   struct Base (virtual base)
  // CHECK-NEXT:         24 |     (Base vftable pointer)
  // CHECK-NEXT:            | [sizeof=32, align=8,
  // CHECK-NEXT:            |  nvsize=16, nvalign=8]
};

int main() {
  Container::f();
  Container2::f1();
  Container2::f2();
  Container3 cont3;
  return 0;
};
