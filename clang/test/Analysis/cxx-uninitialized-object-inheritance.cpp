// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UninitializedObject \
// RUN: -analyzer-config alpha.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN: -analyzer-config alpha.cplusplus.UninitializedObject:CheckPointeeInitialization=true \
// RUN: -std=c++11 -verify  %s

//===----------------------------------------------------------------------===//
// Non-polymorphic inheritance tests
//===----------------------------------------------------------------------===//

class NonPolymorphicLeft1 {
  int x;

protected:
  int y;

public:
  NonPolymorphicLeft1() = default;
  NonPolymorphicLeft1(int) : x(1) {}
};

class NonPolymorphicInheritanceTest1 : public NonPolymorphicLeft1 {
  int z;

public:
  NonPolymorphicInheritanceTest1()
      : NonPolymorphicLeft1(int{}) {
    y = 2;
    z = 3;
    // All good!
  }
};

void fNonPolymorphicInheritanceTest1() {
  NonPolymorphicInheritanceTest1();
}

class NonPolymorphicBaseClass2 {
  int x; // expected-note{{uninitialized field 'this->NonPolymorphicBaseClass2::x'}}
protected:
  int y;

public:
  NonPolymorphicBaseClass2() = default;
  NonPolymorphicBaseClass2(int) : x(4) {}
};

class NonPolymorphicInheritanceTest2 : public NonPolymorphicBaseClass2 {
  int z;

public:
  NonPolymorphicInheritanceTest2() {
    y = 5;
    z = 6; // expected-warning{{1 uninitialized field}}
  }
};

void fNonPolymorphicInheritanceTest2() {
  NonPolymorphicInheritanceTest2();
}

class NonPolymorphicBaseClass3 {
  int x;

protected:
  int y; // expected-note{{uninitialized field 'this->NonPolymorphicBaseClass3::y'}}
public:
  NonPolymorphicBaseClass3() = default;
  NonPolymorphicBaseClass3(int) : x(7) {}
};

class NonPolymorphicInheritanceTest3 : public NonPolymorphicBaseClass3 {
  int z;

public:
  NonPolymorphicInheritanceTest3()
      : NonPolymorphicBaseClass3(int{}) {
    z = 8; // expected-warning{{1 uninitialized field}}
  }
};

void fNonPolymorphicInheritanceTest3() {
  NonPolymorphicInheritanceTest3();
}

class NonPolymorphicBaseClass4 {
  int x;

protected:
  int y;

public:
  NonPolymorphicBaseClass4() = default;
  NonPolymorphicBaseClass4(int) : x(9) {}
};

class NonPolymorphicInheritanceTest4 : public NonPolymorphicBaseClass4 {
  int z; // expected-note{{uninitialized field 'this->z'}}

public:
  NonPolymorphicInheritanceTest4()
      : NonPolymorphicBaseClass4(int{}) {
    y = 10; // expected-warning{{1 uninitialized field}}
  }
};

void fNonPolymorphicInheritanceTest4() {
  NonPolymorphicInheritanceTest4();
}

//===----------------------------------------------------------------------===//
// Polymorphic inheritance tests
//===----------------------------------------------------------------------===//

class PolymorphicLeft1 {
  int x;

protected:
  int y;

public:
  virtual ~PolymorphicLeft1() = default;
  PolymorphicLeft1() = default;
  PolymorphicLeft1(int) : x(11) {}
};

class PolymorphicInheritanceTest1 : public PolymorphicLeft1 {
  int z;

public:
  PolymorphicInheritanceTest1()
      : PolymorphicLeft1(int{}) {
    y = 12;
    z = 13;
    // All good!
  }
};

void fPolymorphicInheritanceTest1() {
  PolymorphicInheritanceTest1();
}

class PolymorphicRight1 {
  int x; // expected-note{{uninitialized field 'this->PolymorphicRight1::x'}}
protected:
  int y;

public:
  virtual ~PolymorphicRight1() = default;
  PolymorphicRight1() = default;
  PolymorphicRight1(int) : x(14) {}
};

class PolymorphicInheritanceTest2 : public PolymorphicRight1 {
  int z;

public:
  PolymorphicInheritanceTest2() {
    y = 15;
    z = 16; // expected-warning{{1 uninitialized field}}
  }
};

void fPolymorphicInheritanceTest2() {
  PolymorphicInheritanceTest2();
}

class PolymorphicBaseClass3 {
  int x;

protected:
  int y; // expected-note{{uninitialized field 'this->PolymorphicBaseClass3::y'}}
public:
  virtual ~PolymorphicBaseClass3() = default;
  PolymorphicBaseClass3() = default;
  PolymorphicBaseClass3(int) : x(17) {}
};

class PolymorphicInheritanceTest3 : public PolymorphicBaseClass3 {
  int z;

public:
  PolymorphicInheritanceTest3()
      : PolymorphicBaseClass3(int{}) {
    z = 18; // expected-warning{{1 uninitialized field}}
  }
};

void fPolymorphicInheritanceTest3() {
  PolymorphicInheritanceTest3();
}

class PolymorphicBaseClass4 {
  int x;

protected:
  int y;

public:
  virtual ~PolymorphicBaseClass4() = default;
  PolymorphicBaseClass4() = default;
  PolymorphicBaseClass4(int) : x(19) {}
};

class PolymorphicInheritanceTest4 : public PolymorphicBaseClass4 {
  int z; // expected-note{{uninitialized field 'this->z'}}

public:
  PolymorphicInheritanceTest4()
      : PolymorphicBaseClass4(int{}) {
    y = 20; // expected-warning{{1 uninitialized field}}
  }
};

void fPolymorphicInheritanceTest4() {
  PolymorphicInheritanceTest4();
}

//===----------------------------------------------------------------------===//
// Virtual inheritance tests
//===----------------------------------------------------------------------===//

class VirtualPolymorphicLeft1 {
  int x;

protected:
  int y;

public:
  virtual ~VirtualPolymorphicLeft1() = default;
  VirtualPolymorphicLeft1() = default;
  VirtualPolymorphicLeft1(int) : x(21) {}
};

class VirtualInheritanceTest1 : virtual public VirtualPolymorphicLeft1 {
  int z;

public:
  VirtualInheritanceTest1()
      : VirtualPolymorphicLeft1(int()) {
    y = 22;
    z = 23;
    // All good!
  }
};

void fVirtualInheritanceTest1() {
  VirtualInheritanceTest1();
}

class VirtualPolymorphicRight1 {
  int x; // expected-note{{uninitialized field 'this->VirtualPolymorphicRight1::x'}}
protected:
  int y;

public:
  virtual ~VirtualPolymorphicRight1() = default;
  VirtualPolymorphicRight1() = default;
  VirtualPolymorphicRight1(int) : x(24) {}
};

class VirtualInheritanceTest2 : virtual public VirtualPolymorphicRight1 {
  int z;

public:
  VirtualInheritanceTest2() {
    y = 25;
    z = 26; // expected-warning{{1 uninitialized field}}
  }
};

void fVirtualInheritanceTest2() {
  VirtualInheritanceTest2();
}

class VirtualPolymorphicBaseClass3 {
  int x;

protected:
  int y; // expected-note{{uninitialized field 'this->VirtualPolymorphicBaseClass3::y'}}
public:
  virtual ~VirtualPolymorphicBaseClass3() = default;
  VirtualPolymorphicBaseClass3() = default;
  VirtualPolymorphicBaseClass3(int) : x(27) {}
};

class VirtualInheritanceTest3 : virtual public VirtualPolymorphicBaseClass3 {
  int z;

public:
  VirtualInheritanceTest3()
      : VirtualPolymorphicBaseClass3(int{}) {
    z = 28; // expected-warning{{1 uninitialized field}}
  }
};

void fVirtualInheritanceTest3() {
  VirtualInheritanceTest3();
}

//===----------------------------------------------------------------------===//
// Multiple inheritance tests
//===----------------------------------------------------------------------===//

/*
        Left        Right
          \           /
           \         /
            \       /
     MultipleInheritanceTest
*/

struct Left1 {
  int x;
  Left1() = default;
  Left1(int) : x(29) {}
};
struct Right1 {
  int y;
  Right1() = default;
  Right1(int) : y(30) {}
};

class MultipleInheritanceTest1 : public Left1, public Right1 {
  int z;

public:
  MultipleInheritanceTest1()
      : Left1(int{}),
        Right1(char{}) {
    z = 31;
    // All good!
  }

  MultipleInheritanceTest1(int)
      : Left1(int{}) {
    y = 32;
    z = 33;
    // All good!
  }

  MultipleInheritanceTest1(int, int)
      : Right1(char{}) {
    x = 34;
    z = 35;
    // All good!
  }
};

void fMultipleInheritanceTest1() {
  MultipleInheritanceTest1();
  MultipleInheritanceTest1(int());
  MultipleInheritanceTest1(int(), int());
}

struct Left2 {
  int x;
  Left2() = default;
  Left2(int) : x(36) {}
};
struct Right2 {
  int y; // expected-note{{uninitialized field 'this->Right2::y'}}
  Right2() = default;
  Right2(int) : y(37) {}
};

class MultipleInheritanceTest2 : public Left2, public Right2 {
  int z;

public:
  MultipleInheritanceTest2()
      : Left2(int{}) {
    z = 38; // expected-warning{{1 uninitialized field}}
  }
};

void fMultipleInheritanceTest2() {
  MultipleInheritanceTest2();
}

struct Left3 {
  int x; // expected-note{{uninitialized field 'this->Left3::x'}}
  Left3() = default;
  Left3(int) : x(39) {}
};
struct Right3 {
  int y;
  Right3() = default;
  Right3(int) : y(40) {}
};

class MultipleInheritanceTest3 : public Left3, public Right3 {
  int z;

public:
  MultipleInheritanceTest3()
      : Right3(char{}) {
    z = 41; // expected-warning{{1 uninitialized field}}
  }
};

void fMultipleInheritanceTest3() {
  MultipleInheritanceTest3();
}

struct Left4 {
  int x;
  Left4() = default;
  Left4(int) : x(42) {}
};
struct Right4 {
  int y;
  Right4() = default;
  Right4(int) : y(43) {}
};

class MultipleInheritanceTest4 : public Left4, public Right4 {
  int z; // expected-note{{uninitialized field 'this->z'}}

public:
  MultipleInheritanceTest4()
      : Left4(int{}),
        Right4(char{}) { // expected-warning{{1 uninitialized field}}
  }
};

void fMultipleInheritanceTest4() {
  MultipleInheritanceTest4();
}

struct Left5 {
  int x;
  Left5() = default;
  Left5(int) : x(44) {}
};
struct Right5 {
  int y; // expected-note{{uninitialized field 'this->Right5::y'}}
  Right5() = default;
  Right5(int) : y(45) {}
};

class MultipleInheritanceTest5 : public Left5, public Right5 {
  int z; // expected-note{{uninitialized field 'this->z'}}

public:
  MultipleInheritanceTest5() // expected-warning{{2 uninitialized fields}}
      : Left5(int{}) {
  }
};

void fMultipleInheritanceTest5() {
  MultipleInheritanceTest5();
}

//===----------------------------------------------------------------------===//
// Non-virtual diamond inheritance tests
//===----------------------------------------------------------------------===//

/*
  NonVirtualBase   NonVirtualBase
        |                |
        |                |
        |                |
     First              Second
        \                /
         \              /
          \            /
  NonVirtualDiamondInheritanceTest
*/

struct NonVirtualBase1 {
  int x;
  NonVirtualBase1() = default;
  NonVirtualBase1(int) : x(46) {}
};
struct First1 : public NonVirtualBase1 {
  First1() = default;
  First1(int) : NonVirtualBase1(int{}) {}
};
struct Second1 : public NonVirtualBase1 {
  Second1() = default;
  Second1(int) : NonVirtualBase1(int{}) {}
};

class NonVirtualDiamondInheritanceTest1 : public First1, public Second1 {
  int z;

public:
  NonVirtualDiamondInheritanceTest1()
      : First1(int{}),
        Second1(int{}) {
    z = 47;
    // All good!
  }

  NonVirtualDiamondInheritanceTest1(int)
      : First1(int{}) {
    Second1::x = 48;
    z = 49;
    // All good!
  }

  NonVirtualDiamondInheritanceTest1(int, int)
      : Second1(int{}) {
    First1::x = 50;
    z = 51;
    // All good!
  }
};

void fNonVirtualDiamondInheritanceTest1() {
  NonVirtualDiamondInheritanceTest1();
  NonVirtualDiamondInheritanceTest1(int());
  NonVirtualDiamondInheritanceTest1(int(), int());
}

struct NonVirtualBase2 {
  int x; // expected-note{{uninitialized field 'this->NonVirtualBase2::x'}}
  NonVirtualBase2() = default;
  NonVirtualBase2(int) : x(52) {}
};
struct First2 : public NonVirtualBase2 {
  First2() = default;
  First2(int) : NonVirtualBase2(int{}) {}
};
struct Second2 : public NonVirtualBase2 {
  Second2() = default;
  Second2(int) : NonVirtualBase2(int{}) {}
};

class NonVirtualDiamondInheritanceTest2 : public First2, public Second2 {
  int z;

public:
  NonVirtualDiamondInheritanceTest2()
      : First2(int{}) {
    z = 53; // expected-warning{{1 uninitialized field}}
  }
};

void fNonVirtualDiamondInheritanceTest2() {
  NonVirtualDiamondInheritanceTest2();
}

struct NonVirtualBase3 {
  int x; // expected-note{{uninitialized field 'this->NonVirtualBase3::x'}}
  NonVirtualBase3() = default;
  NonVirtualBase3(int) : x(54) {}
};
struct First3 : public NonVirtualBase3 {
  First3() = default;
  First3(int) : NonVirtualBase3(int{}) {}
};
struct Second3 : public NonVirtualBase3 {
  Second3() = default;
  Second3(int) : NonVirtualBase3(int{}) {}
};

class NonVirtualDiamondInheritanceTest3 : public First3, public Second3 {
  int z;

public:
  NonVirtualDiamondInheritanceTest3()
      : Second3(int{}) {
    z = 55; // expected-warning{{1 uninitialized field}}
  }
};

void fNonVirtualDiamondInheritanceTest3() {
  NonVirtualDiamondInheritanceTest3();
}

struct NonVirtualBase4 {
  int x; // expected-note{{uninitialized field 'this->NonVirtualBase4::x'}}
  // expected-note@-1{{uninitialized field 'this->NonVirtualBase4::x'}}
  NonVirtualBase4() = default;
  NonVirtualBase4(int) : x(56) {}
};
struct First4 : public NonVirtualBase4 {
  First4() = default;
  First4(int) : NonVirtualBase4(int{}) {}
};
struct Second4 : public NonVirtualBase4 {
  Second4() = default;
  Second4(int) : NonVirtualBase4(int{}) {}
};

class NonVirtualDiamondInheritanceTest4 : public First4, public Second4 {
  int z;

public:
  NonVirtualDiamondInheritanceTest4() {
    z = 57; // expected-warning{{2 uninitialized fields}}
  }
};

void fNonVirtualDiamondInheritanceTest4() {
  NonVirtualDiamondInheritanceTest4();
}

struct NonVirtualBase5 {
  int x;
  NonVirtualBase5() = default;
  NonVirtualBase5(int) : x(58) {}
};
struct First5 : public NonVirtualBase5 {
  First5() = default;
  First5(int) : NonVirtualBase5(int{}) {}
};
struct Second5 : public NonVirtualBase5 {
  Second5() = default;
  Second5(int) : NonVirtualBase5(int{}) {}
};

class NonVirtualDiamondInheritanceTest5 : public First5, public Second5 {
  int z; // expected-note{{uninitialized field 'this->z'}}

public:
  NonVirtualDiamondInheritanceTest5()
      : First5(int{}),
        Second5(int{}) { // expected-warning{{1 uninitialized field}}
  }
};

void fNonVirtualDiamondInheritanceTest5() {
  NonVirtualDiamondInheritanceTest5();
}

struct NonVirtualBase6 {
  int x; // expected-note{{uninitialized field 'this->NonVirtualBase6::x'}}
  NonVirtualBase6() = default;
  NonVirtualBase6(int) : x(59) {}
};
struct First6 : public NonVirtualBase6 {
  First6() = default;
  First6(int) : NonVirtualBase6(int{}) {}
};
struct Second6 : public NonVirtualBase6 {
  Second6() = default;
  Second6(int) : NonVirtualBase6(int{}) {}
};

class NonVirtualDiamondInheritanceTest6 : public First6, public Second6 {
  int z; // expected-note{{uninitialized field 'this->z'}}

public:
  NonVirtualDiamondInheritanceTest6() // expected-warning{{2 uninitialized fields}}
      : First6(int{}) {
    // 'z' and 'Second::x' unintialized
  }
};

void fNonVirtualDiamondInheritanceTest6() {
  NonVirtualDiamondInheritanceTest6();
}

//===----------------------------------------------------------------------===//
// Virtual diamond inheritance tests
//===----------------------------------------------------------------------===//

/*
           VirtualBase
            /       \
           /         \
          /           \
  VirtualFirst     VirtualSecond
          \           /
           \         /
            \       /
   VirtualDiamondInheritanceTest
*/

struct VirtualBase1 {
  int x;
  VirtualBase1() = default;
  VirtualBase1(int) : x(60) {}
};
struct VirtualFirst1 : virtual public VirtualBase1 {
  VirtualFirst1() = default;
  VirtualFirst1(int) : VirtualBase1(int{}) {}
  VirtualFirst1(int, int) { x = 61; }
};
struct VirtualSecond1 : virtual public VirtualBase1 {
  VirtualSecond1() = default;
  VirtualSecond1(int) : VirtualBase1(int{}) {}
  VirtualSecond1(int, int) { x = 62; }
};

class VirtualDiamondInheritanceTest1 : public VirtualFirst1, public VirtualSecond1 {

public:
  VirtualDiamondInheritanceTest1() {
    x = 0;
    // All good!
  }

  VirtualDiamondInheritanceTest1(int)
      : VirtualFirst1(int{}, int{}),
        VirtualSecond1(int{}, int{}) {
    // All good!
  }

  VirtualDiamondInheritanceTest1(int, int)
      : VirtualFirst1(int{}, int{}) {
    // All good!
  }
};

void fVirtualDiamondInheritanceTest1() {
  VirtualDiamondInheritanceTest1();
  VirtualDiamondInheritanceTest1(int());
  VirtualDiamondInheritanceTest1(int(), int());
}

struct VirtualBase2 {
  int x; // expected-note{{uninitialized field 'this->VirtualBase2::x'}}
  VirtualBase2() = default;
  VirtualBase2(int) : x(63) {}
};
struct VirtualFirst2 : virtual public VirtualBase2 {
  VirtualFirst2() = default;
  VirtualFirst2(int) : VirtualBase2(int{}) {}
  VirtualFirst2(int, int) { x = 64; }
};
struct VirtualSecond2 : virtual public VirtualBase2 {
  VirtualSecond2() = default;
  VirtualSecond2(int) : VirtualBase2(int{}) {}
  VirtualSecond2(int, int) { x = 65; }
};

class VirtualDiamondInheritanceTest2 : public VirtualFirst2, public VirtualSecond2 {

public:
  VirtualDiamondInheritanceTest2() // expected-warning{{1 uninitialized field}}
      : VirtualFirst2(int{}) {
    // From the N4659 C++ Standard Working Draft:
    //
    //   (15.6.2.7)
    //   [...] A 'mem-initializer' where the 'mem-initializer-id' denotes a
    //   virtual base class is ignored during execution of a constructor of any
    //   class that is not the most derived class.
    //
    // This means that Left1::x will not be initialized, because in both
    // VirtualFirst::VirtualFirst(int) and VirtualSecond::VirtualSecond(int)
    // the constructor delegation to Left1::Left1(int) will be
    // ignored.
  }
};

void fVirtualDiamondInheritanceTest2() {
  VirtualDiamondInheritanceTest2();
}

struct VirtualBase3 {
  int x; // expected-note{{uninitialized field 'this->VirtualBase3::x'}}
  VirtualBase3() = default;
  VirtualBase3(int) : x(66) {}
};
struct VirtualFirst3 : virtual public VirtualBase3 {
  VirtualFirst3() = default;
  VirtualFirst3(int) : VirtualBase3(int{}) {}
  VirtualFirst3(int, int) { x = 67; }
};
struct VirtualSecond3 : virtual public VirtualBase3 {
  VirtualSecond3() = default;
  VirtualSecond3(int) : VirtualBase3(int{}) {}
  VirtualSecond3(int, int) { x = 68; }
};

class VirtualDiamondInheritanceTest3 : public VirtualFirst3, public VirtualSecond3 {

public:
  VirtualDiamondInheritanceTest3() // expected-warning{{1 uninitialized field}}
      : VirtualFirst3(int{}) {}
};

void fVirtualDiamondInheritanceTest3() {
  VirtualDiamondInheritanceTest3();
}

//===----------------------------------------------------------------------===//
// Dynamic type test.
//===----------------------------------------------------------------------===//

struct DynTBase1 {};
struct DynTDerived1 : DynTBase1 {
  int y; // expected-note{{uninitialized field 'static_cast<struct DynTDerived1 *>(this->bptr)->y'}}
};

struct DynamicTypeTest1 {
  DynTBase1 *bptr;
  int i = 0;

  DynamicTypeTest1(DynTBase1 *bptr) : bptr(bptr) {} // expected-warning{{1 uninitialized field}}
};

void fDynamicTypeTest1() {
  DynTDerived1 d;
  DynamicTypeTest1 t(&d);
};

struct DynTBase2 {
  int x; // expected-note{{uninitialized field 'static_cast<struct DynTDerived2 *>(this->bptr)->DynTBase2::x'}}
};
struct DynTDerived2 : DynTBase2 {
  int y; // expected-note{{uninitialized field 'static_cast<struct DynTDerived2 *>(this->bptr)->y'}}
};

struct DynamicTypeTest2 {
  DynTBase2 *bptr;
  int i = 0;

  DynamicTypeTest2(DynTBase2 *bptr) : bptr(bptr) {} // expected-warning{{2 uninitialized fields}}
};

void fDynamicTypeTest2() {
  DynTDerived2 d;
  DynamicTypeTest2 t(&d);
}

struct SymbolicSuperRegionBase {
  SymbolicSuperRegionBase() {}
};

struct SymbolicSuperRegionDerived : SymbolicSuperRegionBase {
  SymbolicSuperRegionBase *bptr; // no-crash
  SymbolicSuperRegionDerived(SymbolicSuperRegionBase *bptr) : bptr(bptr) {}
};

SymbolicSuperRegionDerived *getSymbolicRegion();

void fSymbolicSuperRegionTest() {
  SymbolicSuperRegionDerived test(getSymbolicRegion());
}
