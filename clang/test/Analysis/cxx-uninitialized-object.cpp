// RUN: %clang_analyze_cc1 -std=c++14 -verify  %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN:   -analyzer-config \
// RUN:     optin.cplusplus.UninitializedObject:CheckPointeeInitialization=true

// RUN: %clang_analyze_cc1 -std=c++14 -verify  %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config \
// RUN:     optin.cplusplus.UninitializedObject:CheckPointeeInitialization=true

//===----------------------------------------------------------------------===//
// Default constructor test.
//===----------------------------------------------------------------------===//

class CompilerGeneratedConstructorTest {
  int a, b, c, d, e, f, g, h, i, j;

public:
  CompilerGeneratedConstructorTest() = default;
};

void fCompilerGeneratedConstructorTest() {
  CompilerGeneratedConstructorTest();
}

#ifdef PEDANTIC
class DefaultConstructorTest {
  int a; // expected-note{{uninitialized field 'this->a'}}

public:
  DefaultConstructorTest();
};

DefaultConstructorTest::DefaultConstructorTest() = default;

void fDefaultConstructorTest() {
  DefaultConstructorTest(); // expected-warning{{1 uninitialized field}}
}
#else
class DefaultConstructorTest {
  int a;

public:
  DefaultConstructorTest();
};

DefaultConstructorTest::DefaultConstructorTest() = default;

void fDefaultConstructorTest() {
  DefaultConstructorTest();
}
#endif // PEDANTIC

//===----------------------------------------------------------------------===//
// Initializer list test.
//===----------------------------------------------------------------------===//

class InitListTest1 {
  int a;
  int b;

public:
  InitListTest1()
      : a(1),
        b(2) {
    // All good!
  }
};

void fInitListTest1() {
  InitListTest1();
}

class InitListTest2 {
  int a;
  int b; // expected-note{{uninitialized field 'this->b'}}

public:
  InitListTest2()
      : a(3) {} // expected-warning{{1 uninitialized field}}
};

void fInitListTest2() {
  InitListTest2();
}

class InitListTest3 {
  int a; // expected-note{{uninitialized field 'this->a'}}
  int b;

public:
  InitListTest3()
      : b(4) {} // expected-warning{{1 uninitialized field}}
};

void fInitListTest3() {
  InitListTest3();
}

//===----------------------------------------------------------------------===//
// Constructor body test.
//===----------------------------------------------------------------------===//

class CtorBodyTest1 {
  int a, b;

public:
  CtorBodyTest1() {
    a = 5;
    b = 6;
    // All good!
  }
};

void fCtorBodyTest1() {
  CtorBodyTest1();
}

class CtorBodyTest2 {
  int a;
  int b; // expected-note{{uninitialized field 'this->b'}}

public:
  CtorBodyTest2() {
    a = 7; // expected-warning{{1 uninitialized field}}
  }
};

void fCtorBodyTest2() {
  CtorBodyTest2();
}

class CtorBodyTest3 {
  int a; // expected-note{{uninitialized field 'this->a'}}
  int b;

public:
  CtorBodyTest3() {
    b = 8; // expected-warning{{1 uninitialized field}}
  }
};

void fCtorBodyTest3() {
  CtorBodyTest3();
}

#ifdef PEDANTIC
class CtorBodyTest4 {
  int a; // expected-note{{uninitialized field 'this->a'}}
  int b; // expected-note{{uninitialized field 'this->b'}}

public:
  CtorBodyTest4() {}
};

void fCtorBodyTest4() {
  CtorBodyTest4(); // expected-warning{{2 uninitialized fields}}
}
#else
class CtorBodyTest4 {
  int a;
  int b;

public:
  CtorBodyTest4() {}
};

void fCtorBodyTest4() {
  CtorBodyTest4();
}
#endif

//===----------------------------------------------------------------------===//
// Constructor delegation test.
//===----------------------------------------------------------------------===//

class CtorDelegationTest1 {
  int a;
  int b;

public:
  CtorDelegationTest1(int)
      : a(9) {
    // leaves 'b' unintialized, but we'll never check this function
  }

  CtorDelegationTest1()
      : CtorDelegationTest1(int{}) { // Initializing 'a'
    b = 10;
    // All good!
  }
};

void fCtorDelegationTest1() {
  CtorDelegationTest1();
}

class CtorDelegationTest2 {
  int a; // expected-note{{uninitialized field 'this->a'}}
  int b;

public:
  CtorDelegationTest2(int)
      : b(11) {
    // leaves 'a' unintialized, but we'll never check this function
  }

  CtorDelegationTest2()
      : CtorDelegationTest2(int{}) { // expected-warning{{1 uninitialized field}}
  }
};

void fCtorDelegationTest2() {
  CtorDelegationTest2();
}

//===----------------------------------------------------------------------===//
// Tests for classes containing records.
//===----------------------------------------------------------------------===//

class ContainsRecordTest1 {
  struct RecordType {
    int x;
    int y;
  } rec;
  int c, d;

public:
  ContainsRecordTest1()
      : rec({12, 13}),
        c(14),
        d(15) {
    // All good!
  }
};

void fContainsRecordTest1() {
  ContainsRecordTest1();
}

class ContainsRecordTest2 {
  struct RecordType {
    int x;
    int y; // expected-note{{uninitialized field 'this->rec.y'}}
  } rec;
  int c, d;

public:
  ContainsRecordTest2()
      : c(16),
        d(17) {
    rec.x = 18; // expected-warning{{1 uninitialized field}}
  }
};

void fContainsRecordTest2() {
  ContainsRecordTest2();
}

class ContainsRecordTest3 {
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->rec.x'}}
    int y; // expected-note{{uninitialized field 'this->rec.y'}}
  } rec;
  int c, d;

public:
  ContainsRecordTest3()
      : c(19),
        d(20) { // expected-warning{{2 uninitialized fields}}
  }
};

void fContainsRecordTest3() {
  ContainsRecordTest3();
}

class ContainsRecordTest4 {
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->rec.x'}}
    int y; // expected-note{{uninitialized field 'this->rec.y'}}
  } rec;
  int c, d; // expected-note{{uninitialized field 'this->d'}}

public:
  ContainsRecordTest4()
      : c(19) { // expected-warning{{3 uninitialized fields}}
  }
};

void fContainsRecordTest4() {
  ContainsRecordTest4();
}

//===----------------------------------------------------------------------===//
// Tests for template classes.
//===----------------------------------------------------------------------===//

template <class T>
class IntTemplateClassTest1 {
  T t;
  int b;

public:
  IntTemplateClassTest1(T i) {
    b = 21;
    t = i;
    // All good!
  }
};

void fIntTemplateClassTest1() {
  IntTemplateClassTest1<int>(22);
}

template <class T>
class IntTemplateClassTest2 {
  T t; // expected-note{{uninitialized field 'this->t'}}
  int b;

public:
  IntTemplateClassTest2() {
    b = 23; // expected-warning{{1 uninitialized field}}
  }
};

void fIntTemplateClassTest2() {
  IntTemplateClassTest2<int>();
}

struct Record {
  int x; // expected-note{{uninitialized field 'this->t.x'}}
  int y; // expected-note{{uninitialized field 'this->t.y'}}
};

template <class T>
class RecordTemplateClassTest {
  T t;
  int b;

public:
  RecordTemplateClassTest() {
    b = 24; // expected-warning{{2 uninitialized fields}}
  }
};

void fRecordTemplateClassTest() {
  RecordTemplateClassTest<Record>();
}

//===----------------------------------------------------------------------===//
// Tests involving functions with unknown implementations.
//===----------------------------------------------------------------------===//

template <class T>
void mayInitialize(T &);

template <class T>
void wontInitialize(const T &);

class PassingToUnknownFunctionTest1 {
  int a, b;

public:
  PassingToUnknownFunctionTest1() {
    mayInitialize(a);
    mayInitialize(b);
    // All good!
  }

  PassingToUnknownFunctionTest1(int) {
    mayInitialize(a);
    // All good!
  }

  PassingToUnknownFunctionTest1(int, int) {
    mayInitialize(*this);
    // All good!
  }
};

void fPassingToUnknownFunctionTest1() {
  PassingToUnknownFunctionTest1();
  PassingToUnknownFunctionTest1(int());
  PassingToUnknownFunctionTest1(int(), int());
}

class PassingToUnknownFunctionTest2 {
  int a; // expected-note{{uninitialized field 'this->a'}}
  int b;

public:
  PassingToUnknownFunctionTest2() {
    wontInitialize(a);
    b = 4; // expected-warning{{1 uninitialized field}}
  }
};

void fPassingToUnknownFunctionTest2() {
  PassingToUnknownFunctionTest2();
}

//===----------------------------------------------------------------------===//
// Tests for classes containing unions.
//===----------------------------------------------------------------------===//

// FIXME: As of writing this checker, there is no good support for union types
// in the Static Analyzer. Here is non-exhaustive list of cases.
// Note that the rules for unions are different in C and C++.
// http://lists.llvm.org/pipermail/cfe-dev/2017-March/052910.html

class ContainsSimpleUnionTest1 {
  union SimpleUnion {
    float uf;
    int ui;
    char uc;
  } u;

public:
  ContainsSimpleUnionTest1() {
    u.uf = 3.14;
    // All good!
  }
};

void fContainsSimpleUnionTest1() {
  ContainsSimpleUnionTest1();
}

class ContainsSimpleUnionTest2 {
  union SimpleUnion {
    float uf;
    int ui;
    char uc;
    // TODO: we'd expect the note: {{uninitialized field 'this->u'}}
  } u; // no-note

public:
  ContainsSimpleUnionTest2() {}
};

void fContainsSimpleUnionTest2() {
  // TODO: we'd expect the warning: {{1 uninitialized field}}
  ContainsSimpleUnionTest2(); // no-warning
}

class UnionPointerTest1 {
public:
  union SimpleUnion {
    float uf;
    int ui;
    char uc;
  };

private:
  SimpleUnion *uptr;

public:
  UnionPointerTest1(SimpleUnion *uptr, int) : uptr(uptr) {
    // All good!
  }
};

void fUnionPointerTest1() {
  UnionPointerTest1::SimpleUnion u;
  u.uf = 41;
  UnionPointerTest1(&u, int());
}

class UnionPointerTest2 {
public:
  union SimpleUnion {
    float uf;
    int ui;
    char uc;
  };

private:
  // TODO: we'd expect the note: {{uninitialized field 'this->uptr'}}
  SimpleUnion *uptr; // no-note

public:
  UnionPointerTest2(SimpleUnion *uptr, char) : uptr(uptr) {}
};

void fUnionPointerTest2() {
  UnionPointerTest2::SimpleUnion u;
  // TODO: we'd expect the warning: {{1 uninitialized field}}
  UnionPointerTest2(&u, int()); // no-warning
}

class ContainsUnionWithRecordTest1 {
  union UnionWithRecord {
    struct RecordType {
      int x;
      int y;
    } us;
    double ud;
    long ul;

    UnionWithRecord(){};
  } u;

public:
  ContainsUnionWithRecordTest1() {
    u.ud = 3.14;
    // All good!
  }
};

void fContainsUnionWithRecordTest1() {
  ContainsUnionWithRecordTest1();
}

class ContainsUnionWithRecordTest2 {
  union UnionWithRecord {
    struct RecordType {
      int x;
      int y;
    } us;
    double ud;
    long ul;

    UnionWithRecord(){};
  } u;

public:
  ContainsUnionWithRecordTest2() {
    u.us = UnionWithRecord::RecordType{42, 43};
    // All good!
  }
};

void fContainsUnionWithRecordTest2() {
  ContainsUnionWithRecordTest1();
}

class ContainsUnionWithRecordTest3 {
  union UnionWithRecord {
    struct RecordType {
      int x;
      int y;
    } us;
    double ud;
    long ul;

    UnionWithRecord(){};
    // TODO: we'd expect the note: {{uninitialized field 'this->u'}}
  } u; // no-note

public:
  ContainsUnionWithRecordTest3() {
    UnionWithRecord::RecordType rec;
    rec.x = 44;
    // TODO: we'd expect the warning: {{1 uninitialized field}}
    u.us = rec; // no-warning
  }
};

void fContainsUnionWithRecordTest3() {
  ContainsUnionWithRecordTest3();
}

class ContainsUnionWithSimpleUnionTest1 {
  union UnionWithSimpleUnion {
    union SimpleUnion {
      float uf;
      int ui;
      char uc;
    } usu;
    long ul;
    unsigned uu;
  } u;

public:
  ContainsUnionWithSimpleUnionTest1() {
    u.usu.ui = 5;
    // All good!
  }
};

void fContainsUnionWithSimpleUnionTest1() {
  ContainsUnionWithSimpleUnionTest1();
}

class ContainsUnionWithSimpleUnionTest2 {
  union UnionWithSimpleUnion {
    union SimpleUnion {
      float uf;
      int ui;
      char uc;
    } usu;
    long ul;
    unsigned uu;
    // TODO: we'd expect the note: {{uninitialized field 'this->u'}}
  } u; // no-note

public:
  ContainsUnionWithSimpleUnionTest2() {}
};

void fContainsUnionWithSimpleUnionTest2() {
  // TODO: we'd expect the warning: {{1 uninitialized field}}
  ContainsUnionWithSimpleUnionTest2(); // no-warning
}

//===----------------------------------------------------------------------===//
// Zero initialization tests.
//===----------------------------------------------------------------------===//

struct GlobalVariableTest {
  int i;

  GlobalVariableTest() {}
};

GlobalVariableTest gvt; // no-warning

//===----------------------------------------------------------------------===//
// Copy and move constructor tests.
//===----------------------------------------------------------------------===//

template <class T>
void funcToSquelchCompilerWarnings(const T &t);

#ifdef PEDANTIC
struct CopyConstructorTest {
  int i; // expected-note{{uninitialized field 'this->i'}}

  CopyConstructorTest() : i(1337) {}
  CopyConstructorTest(const CopyConstructorTest &other) {}
};

void fCopyConstructorTest() {
  CopyConstructorTest cct;
  CopyConstructorTest copy = cct; // expected-warning{{1 uninitialized field}}
  funcToSquelchCompilerWarnings(copy);
}
#else
struct CopyConstructorTest {
  int i;

  CopyConstructorTest() : i(1337) {}
  CopyConstructorTest(const CopyConstructorTest &other) {}
};

void fCopyConstructorTest() {
  CopyConstructorTest cct;
  CopyConstructorTest copy = cct;
  funcToSquelchCompilerWarnings(copy);
}
#endif // PEDANTIC

struct MoveConstructorTest {
  // TODO: we'd expect the note: {{uninitialized field 'this->i'}}
  int i; // no-note

  MoveConstructorTest() : i(1337) {}
  MoveConstructorTest(const CopyConstructorTest &other) = delete;
  MoveConstructorTest(const CopyConstructorTest &&other) {}
};

void fMoveConstructorTest() {
  MoveConstructorTest cct;
  // TODO: we'd expect the warning: {{1 uninitialized field}}
  MoveConstructorTest copy(static_cast<MoveConstructorTest &&>(cct)); // no-warning
  funcToSquelchCompilerWarnings(copy);
}

//===----------------------------------------------------------------------===//
// Array tests.
//===----------------------------------------------------------------------===//

struct IntArrayTest {
  int arr[256];

  IntArrayTest() {
    // All good!
  }
};

void fIntArrayTest() {
  IntArrayTest();
}

struct RecordTypeArrayTest {
  struct RecordType {
    int x, y;
  } arr[256];

  RecordTypeArrayTest() {
    // All good!
  }
};

void fRecordTypeArrayTest() {
  RecordTypeArrayTest();
}

template <class T>
class CharArrayPointerTest {
  T *t; // no-crash

public:
  CharArrayPointerTest(T *t, int) : t(t) {}
};

void fCharArrayPointerTest() {
  char str[16] = "012345678912345";
  CharArrayPointerTest<char[16]>(&str, int());
}

//===----------------------------------------------------------------------===//
// Memset tests.
//===----------------------------------------------------------------------===//

struct MemsetTest1 {
  int a, b, c;

  MemsetTest1() {
    __builtin_memset(this, 0, sizeof(decltype(*this)));
  }
};

void fMemsetTest1() {
  MemsetTest1();
}

struct MemsetTest2 {
  int a;

  MemsetTest2() {
    __builtin_memset(&a, 0, sizeof(int));
  }
};

void fMemsetTest2() {
  MemsetTest2();
}

//===----------------------------------------------------------------------===//
// Lambda tests.
//===----------------------------------------------------------------------===//

template <class Callable>
struct LambdaThisTest {
  Callable functor;

  LambdaThisTest(const Callable &functor, int) : functor(functor) {
    // All good!
  }
};

struct HasCapturableThis {
  void fLambdaThisTest() {
    auto isEven = [this](int a) { return a % 2 == 0; }; // no-crash
    LambdaThisTest<decltype(isEven)>(isEven, int());
  }
};

template <class Callable>
struct LambdaTest1 {
  Callable functor;

  LambdaTest1(const Callable &functor, int) : functor(functor) {
    // All good!
  }
};

void fLambdaTest1() {
  auto isEven = [](int a) { return a % 2 == 0; };
  LambdaTest1<decltype(isEven)>(isEven, int());
}

#ifdef PEDANTIC
template <class Callable>
struct LambdaTest2 {
  Callable functor;

  LambdaTest2(const Callable &functor, int) : functor(functor) {} // expected-warning{{1 uninitialized field}}
};

void fLambdaTest2() {
  int b;
  auto equals = [&b](int a) { return a == b; }; // expected-note{{uninitialized pointee 'this->functor./*captured variable*/b'}}
  LambdaTest2<decltype(equals)>(equals, int());
}
#else
template <class Callable>
struct LambdaTest2 {
  Callable functor;

  LambdaTest2(const Callable &functor, int) : functor(functor) {}
};

void fLambdaTest2() {
  int b;
  auto equals = [&b](int a) { return a == b; };
  LambdaTest2<decltype(equals)>(equals, int());
}
#endif //PEDANTIC

#ifdef PEDANTIC
namespace LT3Detail {

struct RecordType {
  int x; // expected-note{{uninitialized field 'this->functor./*captured variable*/rec1.x'}}
  int y; // expected-note{{uninitialized field 'this->functor./*captured variable*/rec1.y'}}
};

} // namespace LT3Detail
template <class Callable>
struct LambdaTest3 {
  Callable functor;

  LambdaTest3(const Callable &functor, int) : functor(functor) {} // expected-warning{{2 uninitialized fields}}
};

void fLambdaTest3() {
  LT3Detail::RecordType rec1;
  auto equals = [&rec1](LT3Detail::RecordType rec2) {
    return rec1.x == rec2.x;
  };
  LambdaTest3<decltype(equals)>(equals, int());
}
#else
namespace LT3Detail {

struct RecordType {
  int x;
  int y;
};

} // namespace LT3Detail
template <class Callable>
struct LambdaTest3 {
  Callable functor;

  LambdaTest3(const Callable &functor, int) : functor(functor) {}
};

void fLambdaTest3() {
  LT3Detail::RecordType rec1;
  auto equals = [&rec1](LT3Detail::RecordType rec2) {
    return rec1.x == rec2.x;
  };
  LambdaTest3<decltype(equals)>(equals, int());
}
#endif //PEDANTIC

template <class Callable>
struct MultipleLambdaCapturesTest1 {
  Callable functor;
  int dontGetFilteredByNonPedanticMode = 0;

  MultipleLambdaCapturesTest1(const Callable &functor, int) : functor(functor) {} // expected-warning{{2 uninitialized field}}
};

void fMultipleLambdaCapturesTest1() {
  int b1, b2 = 3, b3;
  auto equals = [&b1, &b2, &b3](int a) { return a == b1 == b2 == b3; }; // expected-note{{uninitialized pointee 'this->functor./*captured variable*/b1'}}
  // expected-note@-1{{uninitialized pointee 'this->functor./*captured variable*/b3'}}
  MultipleLambdaCapturesTest1<decltype(equals)>(equals, int());
}

template <class Callable>
struct MultipleLambdaCapturesTest2 {
  Callable functor;
  int dontGetFilteredByNonPedanticMode = 0;

  MultipleLambdaCapturesTest2(const Callable &functor, int) : functor(functor) {} // expected-warning{{1 uninitialized field}}
};

void fMultipleLambdaCapturesTest2() {
  int b1, b2 = 3, b3;
  auto equals = [b1, &b2, &b3](int a) { return a == b1 == b2 == b3; }; // expected-note{{uninitialized pointee 'this->functor./*captured variable*/b3'}}
  MultipleLambdaCapturesTest2<decltype(equals)>(equals, int());
}

struct LambdaWrapper {
  void *func; // no-crash
  int dontGetFilteredByNonPedanticMode = 0;

  LambdaWrapper(void *ptr) : func(ptr) {} // expected-warning{{1 uninitialized field}}
};

struct ThisCapturingLambdaFactory {
  int a; // expected-note{{uninitialized field 'static_cast<decltype(a.ret()) *>(this->func)->/*'this' capture*/->a'}}

  auto ret() {
    return [this] { (void)this; };
  }
};

void fLambdaFieldWithInvalidThisCapture() {
  void *ptr;
  {
    ThisCapturingLambdaFactory a;
    decltype(a.ret()) lambda = a.ret();
    ptr = &lambda;
  }
  LambdaWrapper t(ptr);
}

//===----------------------------------------------------------------------===//
// System header tests.
//===----------------------------------------------------------------------===//

#include "Inputs/system-header-simulator-for-cxx-uninitialized-object.h"

struct SystemHeaderTest1 {
  RecordInSystemHeader rec; // defined in the system header simulator

  SystemHeaderTest1() {
    // All good!
  }
};

void fSystemHeaderTest1() {
  SystemHeaderTest1();
}

#ifdef PEDANTIC
struct SystemHeaderTest2 {
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->container.t.x}}
    int y; // expected-note{{uninitialized field 'this->container.t.y}}
  };
  ContainerInSystemHeader<RecordType> container;

  SystemHeaderTest2(RecordType &rec, int) : container(rec) {} // expected-warning{{2 uninitialized fields}}
};

void fSystemHeaderTest2() {
  SystemHeaderTest2::RecordType rec;
  SystemHeaderTest2(rec, int());
}
#else
struct SystemHeaderTest2 {
  struct RecordType {
    int x;
    int y;
  };
  ContainerInSystemHeader<RecordType> container;

  SystemHeaderTest2(RecordType &rec, int) : container(rec) {}
};

void fSystemHeaderTest2() {
  SystemHeaderTest2::RecordType rec;
  SystemHeaderTest2(rec, int());
}
#endif //PEDANTIC

//===----------------------------------------------------------------------===//
// Incomplete type tests.
//===----------------------------------------------------------------------===//

struct IncompleteTypeTest1 {
  struct RecordType;
  // no-crash
  RecordType *recptr; // expected-note{{uninitialized pointer 'this->recptr}}
  int dontGetFilteredByNonPedanticMode = 0;

  IncompleteTypeTest1() {} // expected-warning{{1 uninitialized field}}
};

void fIncompleteTypeTest1() {
  IncompleteTypeTest1();
}

struct IncompleteTypeTest2 {
  struct RecordType;
  RecordType *recptr; // no-crash
  int dontGetFilteredByNonPedanticMode = 0;

  RecordType *recordTypeFactory();

  IncompleteTypeTest2() : recptr(recordTypeFactory()) {}
};

void fIncompleteTypeTest2() {
  IncompleteTypeTest2();
}

struct IncompleteTypeTest3 {
  struct RecordType;
  RecordType &recref; // no-crash
  int dontGetFilteredByNonPedanticMode = 0;

  RecordType &recordTypeFactory();

  IncompleteTypeTest3() : recref(recordTypeFactory()) {}
};

void fIncompleteTypeTest3() {
  IncompleteTypeTest3();
}

//===----------------------------------------------------------------------===//
// Builtin type or enumeration type related tests.
//===----------------------------------------------------------------------===//

struct IntegralTypeTest {
  int a; // expected-note{{uninitialized field 'this->a'}}
  int dontGetFilteredByNonPedanticMode = 0;

  IntegralTypeTest() {} // expected-warning{{1 uninitialized field}}
};

void fIntegralTypeTest() {
  IntegralTypeTest();
}

struct FloatingTypeTest {
  float a; // expected-note{{uninitialized field 'this->a'}}
  int dontGetFilteredByNonPedanticMode = 0;

  FloatingTypeTest() {} // expected-warning{{1 uninitialized field}}
};

void fFloatingTypeTest() {
  FloatingTypeTest();
}

struct NullptrTypeTypeTest {
  decltype(nullptr) a; // expected-note{{uninitialized field 'this->a'}}
  int dontGetFilteredByNonPedanticMode = 0;

  NullptrTypeTypeTest() {} // expected-warning{{1 uninitialized field}}
};

void fNullptrTypeTypeTest() {
  NullptrTypeTypeTest();
}

struct EnumTest {
  enum Enum {
    A,
    B
  } enum1; // expected-note{{uninitialized field 'this->enum1'}}
  enum class Enum2 {
    A,
    B
  } enum2; // expected-note{{uninitialized field 'this->enum2'}}
  int dontGetFilteredByNonPedanticMode = 0;

  EnumTest() {} // expected-warning{{2 uninitialized fields}}
};

void fEnumTest() {
  EnumTest();
}

//===----------------------------------------------------------------------===//
// Tests for constructor calls within another cunstructor, without the two
// records being in any relation.
//===----------------------------------------------------------------------===//

void halt() __attribute__((__noreturn__));
void assert(int b) {
  if (!b)
    halt();
}

// While a singleton would make more sense as a static variable, that would zero
// initialize all of its fields, hence the not too practical implementation.
struct Singleton {
  int i; // expected-note{{uninitialized field 'this->i'}}
  int dontGetFilteredByNonPedanticMode = 0;

  Singleton() {
    assert(!isInstantiated);
    isInstantiated = true; // expected-warning{{1 uninitialized field}}
  }

  ~Singleton() {
    isInstantiated = false;
  }

  static bool isInstantiated;
};

bool Singleton::isInstantiated = false;

struct SingletonTest {
  int dontGetFilteredByNonPedanticMode = 0;

  SingletonTest() {
    Singleton();
  }
};

void fSingletonTest() {
  SingletonTest();
}

//===----------------------------------------------------------------------===//
// C++11 member initializer tests.
//===----------------------------------------------------------------------===//

struct CXX11MemberInitTest1 {
  int a = 3;
  int b;
  CXX11MemberInitTest1() : b(2) {
    // All good!
  }
};

void fCXX11MemberInitTest1() {
  CXX11MemberInitTest1();
}

#ifdef PEDANTIC
struct CXX11MemberInitTest2 {
  struct RecordType {
    int a; // expected-note {{uninitialized field 'this->a'}}
    int b; // expected-note {{uninitialized field 'this->b'}}

    RecordType(int) {}
  };

  RecordType rec = RecordType(int()); // expected-warning {{2 uninitialized fields}}
  int dontGetFilteredByNonPedanticMode = 0;

  CXX11MemberInitTest2() {}
};

void fCXX11MemberInitTest2() {
  CXX11MemberInitTest2(); // no-warning
}

#endif // PEDANTIC

//===----------------------------------------------------------------------===//
// "Esoteric" primitive type tests.
//===----------------------------------------------------------------------===//

struct MyAtomicInt {
  _Atomic(int) x; // expected-note{{uninitialized field 'this->x'}}
  int dontGetFilteredByNonPedanticMode = 0;

  MyAtomicInt() {} // expected-warning{{1 uninitialized field}}
};

void _AtomicTest() {
  MyAtomicInt b;
}

struct VectorSizeLong {
  VectorSizeLong() {}
  __attribute__((__vector_size__(16))) long x;
};

void __vector_size__LongTest() {
  // TODO: Warn for v.x.
  VectorSizeLong v;
  v.x[0] = 0;
}

struct ComplexUninitTest {
  ComplexUninitTest() {}
  __complex__ float x;
  __complex__ int y;
};

struct ComplexInitTest {
  ComplexInitTest() {
    x = {1.0f, 1.0f};
    y = {1, 1};
  }
  __complex__ float x;
  __complex__ int y;
};

void fComplexTest() {
  ComplexInitTest x;

  // TODO: we should emit a warning for x2.x and x2.y.
  ComplexUninitTest x2;
}
