// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UninitializedObject \
// RUN:   -analyzer-config alpha.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN:   -analyzer-config alpha.cplusplus.UninitializedObject:CheckPointeeInitialization=true \
// RUN:   -std=c++11 -verify  %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UninitializedObject \
// RUN:   -analyzer-config alpha.cplusplus.UninitializedObject:CheckPointeeInitialization=true \
// RUN:   -std=c++11 -verify  %s

//===----------------------------------------------------------------------===//
// Concrete location tests.
//===----------------------------------------------------------------------===//

struct ConcreteIntLocTest {
  int *ptr;

  ConcreteIntLocTest() : ptr(reinterpret_cast<int *>(0xDEADBEEF)) {}
};

void fConcreteIntLocTest() {
  ConcreteIntLocTest();
}

//===----------------------------------------------------------------------===//
// nonloc::LocAsInteger tests.
//===----------------------------------------------------------------------===//

using intptr_t = unsigned long long;

struct LocAsIntegerTest {
  intptr_t ptr; // expected-note{{uninitialized pointee 'reinterpret_cast<char *>(this->ptr)'}}
  int dontGetFilteredByNonPedanticMode = 0;

  LocAsIntegerTest(void *ptr) : ptr(reinterpret_cast<intptr_t>(ptr)) {} // expected-warning{{1 uninitialized field}}
};

void fLocAsIntegerTest() {
  char c;
  LocAsIntegerTest t(&c);
}

//===----------------------------------------------------------------------===//
// Null pointer tests.
//===----------------------------------------------------------------------===//

class NullPtrTest {
  struct RecordType {
    int x;
    int y;
  };

  float *fptr = nullptr;
  int *ptr;
  RecordType *recPtr;

public:
  NullPtrTest() : ptr(nullptr), recPtr(nullptr) {
    // All good!
  }
};

void fNullPtrTest() {
  NullPtrTest();
}

//===----------------------------------------------------------------------===//
// Alloca tests.
//===----------------------------------------------------------------------===//

struct UntypedAllocaTest {
  void *allocaPtr;
  int dontGetFilteredByNonPedanticMode = 0;

  UntypedAllocaTest() : allocaPtr(__builtin_alloca(sizeof(int))) {
    // All good!
  }
};

void fUntypedAllocaTest() {
  UntypedAllocaTest();
}

struct TypedAllocaTest1 {
  int *allocaPtr; // expected-note{{uninitialized pointee 'this->allocaPtr'}}
  int dontGetFilteredByNonPedanticMode = 0;

  TypedAllocaTest1() // expected-warning{{1 uninitialized field}}
      : allocaPtr(static_cast<int *>(__builtin_alloca(sizeof(int)))) {}
};

void fTypedAllocaTest1() {
  TypedAllocaTest1();
}

struct TypedAllocaTest2 {
  int *allocaPtr;
  int dontGetFilteredByNonPedanticMode = 0;

  TypedAllocaTest2()
      : allocaPtr(static_cast<int *>(__builtin_alloca(sizeof(int)))) {
    *allocaPtr = 55555;
    // All good!
  }
};

void fTypedAllocaTest2() {
  TypedAllocaTest2();
}

//===----------------------------------------------------------------------===//
// Heap pointer tests.
//===----------------------------------------------------------------------===//

class HeapPointerTest1 {
  struct RecordType {
    // TODO: we'd expect the note: {{uninitialized field 'this->recPtr->y'}}
    int x; // no-note
    // TODO: we'd expect the note: {{uninitialized field 'this->recPtr->y'}}
    int y; // no-note
  };
  // TODO: we'd expect the note: {{uninitialized pointee 'this->fptr'}}
  float *fptr = new float; // no-note
  // TODO: we'd expect the note: {{uninitialized pointee 'this->ptr'}}
  int *ptr; // no-note
  RecordType *recPtr;

public:
  // TODO: we'd expect the warning: {{4 uninitialized fields}}
  HeapPointerTest1() : ptr(new int), recPtr(new RecordType) { // no-note
  }
};

void fHeapPointerTest1() {
  HeapPointerTest1();
}

class HeapPointerTest2 {
  struct RecordType {
    int x;
    int y;
  };

  float *fptr = new float(); // initializes to 0
  int *ptr;
  RecordType *recPtr;

public:
  HeapPointerTest2() : ptr(new int{25}), recPtr(new RecordType{26, 27}) {
    // All good!
  }
};

void fHeapPointerTest2() {
  HeapPointerTest2();
}

//===----------------------------------------------------------------------===//
// Stack pointer tests.
//===----------------------------------------------------------------------===//

class StackPointerTest1 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  int *ptr;
  RecordType *recPtr;

public:
  StackPointerTest1(int *_ptr, StackPointerTest1::RecordType *_recPtr) : ptr(_ptr), recPtr(_recPtr) {
    // All good!
  }
};

void fStackPointerTest1() {
  int ok_a = 28;
  StackPointerTest1::RecordType ok_rec{29, 30};
  StackPointerTest1(&ok_a, &ok_rec); // 'a', 'rec.x', 'rec.y' uninitialized
}

#ifdef PEDANTIC
class StackPointerTest2 {
public:
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->recPtr->x'}}
    int y; // expected-note{{uninitialized field 'this->recPtr->y'}}
  };

private:
  int *ptr; // expected-note{{uninitialized pointee 'this->ptr'}}
  RecordType *recPtr;

public:
  StackPointerTest2(int *_ptr, RecordType *_recPtr) : ptr(_ptr), recPtr(_recPtr) { // expected-warning{{3 uninitialized fields}}
  }
};

void fStackPointerTest2() {
  int a;
  StackPointerTest2::RecordType rec;
  StackPointerTest2(&a, &rec); // 'a', 'rec.x', 'rec.y' uninitialized
}
#else
class StackPointerTest2 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  int *ptr;
  RecordType *recPtr;

public:
  StackPointerTest2(int *_ptr, RecordType *_recPtr) : ptr(_ptr), recPtr(_recPtr) {
  }
};

void fStackPointerTest2() {
  int a;
  StackPointerTest2::RecordType rec;
  StackPointerTest2(&a, &rec); // 'a', 'rec.x', 'rec.y' uninitialized
}
#endif // PEDANTIC

class UninitPointerTest {
  struct RecordType {
    int x;
    int y;
  };

  int *ptr; // expected-note{{uninitialized pointer 'this->ptr'}}
  RecordType *recPtr;

public:
  UninitPointerTest() : recPtr(new RecordType{13, 13}) { // expected-warning{{1 uninitialized field}}
  }
};

void fUninitPointerTest() {
  UninitPointerTest();
}

struct CharPointerTest {
  const char *str;
  int dontGetFilteredByNonPedanticMode = 0;

  CharPointerTest() : str("") {}
};

void fCharPointerTest() {
  CharPointerTest();
}

struct CyclicPointerTest1 {
  int *ptr; // expected-note{{object references itself 'this->ptr'}}
  int dontGetFilteredByNonPedanticMode = 0;

  CyclicPointerTest1() : ptr(reinterpret_cast<int *>(&ptr)) {} // expected-warning{{1 uninitialized field}}
};

void fCyclicPointerTest1() {
  CyclicPointerTest1();
}

struct CyclicPointerTest2 {
  int **pptr; // expected-note{{object references itself 'this->pptr'}}
  int dontGetFilteredByNonPedanticMode = 0;

  CyclicPointerTest2() : pptr(reinterpret_cast<int **>(&pptr)) {} // expected-warning{{1 uninitialized field}}
};

void fCyclicPointerTest2() {
  CyclicPointerTest2();
}

//===----------------------------------------------------------------------===//
// Void pointer tests.
//===----------------------------------------------------------------------===//

// Void pointer tests are mainly no-crash tests.

void *malloc(int size);

class VoidPointerTest1 {
  void *vptr;

public:
  VoidPointerTest1(void *vptr, char) : vptr(vptr) {
    // All good!
  }
};

void fVoidPointerTest1() {
  void *vptr = malloc(sizeof(int));
  VoidPointerTest1(vptr, char());
}

class VoidPointerTest2 {
  void **vpptr;

public:
  VoidPointerTest2(void **vpptr, char) : vpptr(vpptr) {
    // All good!
  }
};

void fVoidPointerTest2() {
  void *vptr = malloc(sizeof(int));
  VoidPointerTest2(&vptr, char());
}

class VoidPointerRRefTest1 {
  void *&&vptrrref; // expected-note {{here}}

public:
  VoidPointerRRefTest1(void *vptr, char) : vptrrref(static_cast<void *&&>(vptr)) { // expected-warning {{binding reference member 'vptrrref' to stack allocated parameter 'vptr'}}
    // All good!
  }
};

void fVoidPointerRRefTest1() {
  void *vptr = malloc(sizeof(int));
  VoidPointerRRefTest1(vptr, char());
}

class VoidPointerRRefTest2 {
  void **&&vpptrrref; // expected-note {{here}}

public:
  VoidPointerRRefTest2(void **vptr, char) : vpptrrref(static_cast<void **&&>(vptr)) { // expected-warning {{binding reference member 'vpptrrref' to stack allocated parameter 'vptr'}}
    // All good!
  }
};

void fVoidPointerRRefTest2() {
  void *vptr = malloc(sizeof(int));
  VoidPointerRRefTest2(&vptr, char());
}

class VoidPointerLRefTest {
  void *&vptrrref; // expected-note {{here}}

public:
  VoidPointerLRefTest(void *vptr, char) : vptrrref(static_cast<void *&>(vptr)) { // expected-warning {{binding reference member 'vptrrref' to stack allocated parameter 'vptr'}}
    // All good!
  }
};

void fVoidPointerLRefTest() {
  void *vptr = malloc(sizeof(int));
  VoidPointerLRefTest(vptr, char());
}

struct CyclicVoidPointerTest {
  void *vptr; // expected-note{{object references itself 'this->vptr'}}
  int dontGetFilteredByNonPedanticMode = 0;

  CyclicVoidPointerTest() : vptr(&vptr) {} // expected-warning{{1 uninitialized field}}
};

void fCyclicVoidPointerTest() {
  CyclicVoidPointerTest();
}

struct IntDynTypedVoidPointerTest1 {
  void *vptr; // expected-note{{uninitialized pointee 'static_cast<int *>(this->vptr)'}}
  int dontGetFilteredByNonPedanticMode = 0;

  IntDynTypedVoidPointerTest1(void *vptr) : vptr(vptr) {} // expected-warning{{1 uninitialized field}}
};

void fIntDynTypedVoidPointerTest1() {
  int a;
  IntDynTypedVoidPointerTest1 tmp(&a);
}

struct RecordDynTypedVoidPointerTest {
  struct RecordType {
    int x; // expected-note{{uninitialized field 'static_cast<struct RecordDynTypedVoidPointerTest::RecordType *>(this->vptr)->x'}}
    int y; // expected-note{{uninitialized field 'static_cast<struct RecordDynTypedVoidPointerTest::RecordType *>(this->vptr)->y'}}
  };

  void *vptr;
  int dontGetFilteredByNonPedanticMode = 0;

  RecordDynTypedVoidPointerTest(void *vptr) : vptr(vptr) {} // expected-warning{{2 uninitialized fields}}
};

void fRecordDynTypedVoidPointerTest() {
  RecordDynTypedVoidPointerTest::RecordType a;
  RecordDynTypedVoidPointerTest tmp(&a);
}

struct NestedNonVoidDynTypedVoidPointerTest {
  struct RecordType {
    int x;      // expected-note{{uninitialized field 'static_cast<struct NestedNonVoidDynTypedVoidPointerTest::RecordType *>(this->vptr)->x'}}
    int y;      // expected-note{{uninitialized field 'static_cast<struct NestedNonVoidDynTypedVoidPointerTest::RecordType *>(this->vptr)->y'}}
    void *vptr; // expected-note{{uninitialized pointee 'static_cast<char *>(static_cast<struct NestedNonVoidDynTypedVoidPointerTest::RecordType *>(this->vptr)->vptr)'}}
  };

  void *vptr;
  int dontGetFilteredByNonPedanticMode = 0;

  NestedNonVoidDynTypedVoidPointerTest(void *vptr, void *c) : vptr(vptr) {
    static_cast<RecordType *>(vptr)->vptr = c; // expected-warning{{3 uninitialized fields}}
  }
};

void fNestedNonVoidDynTypedVoidPointerTest() {
  NestedNonVoidDynTypedVoidPointerTest::RecordType a;
  char c;
  NestedNonVoidDynTypedVoidPointerTest tmp(&a, &c);
}

//===----------------------------------------------------------------------===//
// Multipointer tests.
//===----------------------------------------------------------------------===//

#ifdef PEDANTIC
class MultiPointerTest1 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  RecordType **mptr; // expected-note{{uninitialized pointee 'this->mptr'}}

public:
  MultiPointerTest1(RecordType **p, int) : mptr(p) { // expected-warning{{1 uninitialized field}}
  }
};

void fMultiPointerTest1() {
  MultiPointerTest1::RecordType *p1;
  MultiPointerTest1::RecordType **mptr = &p1;
  MultiPointerTest1(mptr, int()); // '*mptr' uninitialized
}
#else
class MultiPointerTest1 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  RecordType **mptr;

public:
  MultiPointerTest1(RecordType **p, int) : mptr(p) {}
};

void fMultiPointerTest1() {
  MultiPointerTest1::RecordType *p1;
  MultiPointerTest1::RecordType **mptr = &p1;
  MultiPointerTest1(mptr, int()); // '*mptr' uninitialized
}
#endif // PEDANTIC

#ifdef PEDANTIC
class MultiPointerTest2 {
public:
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->mptr->x'}}
    int y; // expected-note{{uninitialized field 'this->mptr->y'}}
  };

private:
  RecordType **mptr;

public:
  MultiPointerTest2(RecordType **p, int) : mptr(p) { // expected-warning{{2 uninitialized fields}}
  }
};

void fMultiPointerTest2() {
  MultiPointerTest2::RecordType i;
  MultiPointerTest2::RecordType *p1 = &i;
  MultiPointerTest2::RecordType **mptr = &p1;
  MultiPointerTest2(mptr, int()); // '**mptr' uninitialized
}
#else
class MultiPointerTest2 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  RecordType **mptr;

public:
  MultiPointerTest2(RecordType **p, int) : mptr(p) {
  }
};

void fMultiPointerTest2() {
  MultiPointerTest2::RecordType i;
  MultiPointerTest2::RecordType *p1 = &i;
  MultiPointerTest2::RecordType **mptr = &p1;
  MultiPointerTest2(mptr, int()); // '**mptr' uninitialized
}
#endif // PEDANTIC

class MultiPointerTest3 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  RecordType **mptr;

public:
  MultiPointerTest3(RecordType **p, int) : mptr(p) {
    // All good!
  }
};

void fMultiPointerTest3() {
  MultiPointerTest3::RecordType i{31, 32};
  MultiPointerTest3::RecordType *p1 = &i;
  MultiPointerTest3::RecordType **mptr = &p1;
  MultiPointerTest3(mptr, int()); // '**mptr' uninitialized
}

//===----------------------------------------------------------------------===//
// Incomplete pointee tests.
//===----------------------------------------------------------------------===//

class IncompleteType;

struct IncompletePointeeTypeTest {
  IncompleteType *pImpl; //no-crash
  int dontGetFilteredByNonPedanticMode = 0;

  IncompletePointeeTypeTest(IncompleteType *A) : pImpl(A) {}
};

void fIncompletePointeeTypeTest(void *ptr) {
  IncompletePointeeTypeTest(reinterpret_cast<IncompleteType *>(ptr));
}

//===----------------------------------------------------------------------===//
// Function pointer tests.
//===----------------------------------------------------------------------===//

struct FunctionPointerWithDifferentDynTypeTest {
  using Func1 = void *(*)();
  using Func2 = int *(*)();

  Func1 f; // no-crash
  FunctionPointerWithDifferentDynTypeTest(Func2 f) : f((Func1)f) {}
};

// Note that there isn't a function calling the constructor of
// FunctionPointerWithDifferentDynTypeTest, because a crash could only be
// reproduced without it.

//===----------------------------------------------------------------------===//
// Member pointer tests.
//===----------------------------------------------------------------------===//

struct UsefulFunctions {
  int a, b;

  void print() {}
  void dump() {}
};

#ifdef PEDANTIC
struct PointerToMemberFunctionTest1 {
  void (UsefulFunctions::*f)(void); // expected-note{{uninitialized field 'this->f'}}
  PointerToMemberFunctionTest1() {}
};

void fPointerToMemberFunctionTest1() {
  PointerToMemberFunctionTest1(); // expected-warning{{1 uninitialized field}}
}

struct PointerToMemberFunctionTest2 {
  void (UsefulFunctions::*f)(void);
  PointerToMemberFunctionTest2(void (UsefulFunctions::*f)(void)) : f(f) {
    // All good!
  }
};

void fPointerToMemberFunctionTest2() {
  void (UsefulFunctions::*f)(void) = &UsefulFunctions::print;
  PointerToMemberFunctionTest2 a(f);
}

struct MultiPointerToMemberFunctionTest1 {
  void (UsefulFunctions::**f)(void); // expected-note{{uninitialized pointer 'this->f'}}
  MultiPointerToMemberFunctionTest1() {}
};

void fMultiPointerToMemberFunctionTest1() {
  MultiPointerToMemberFunctionTest1(); // expected-warning{{1 uninitialized field}}
}

struct MultiPointerToMemberFunctionTest2 {
  void (UsefulFunctions::**f)(void);
  MultiPointerToMemberFunctionTest2(void (UsefulFunctions::**f)(void)) : f(f) {
    // All good!
  }
};

void fMultiPointerToMemberFunctionTest2() {
  void (UsefulFunctions::*f)(void) = &UsefulFunctions::print;
  MultiPointerToMemberFunctionTest2 a(&f);
}

struct PointerToMemberDataTest1 {
  int UsefulFunctions::*d; // expected-note{{uninitialized field 'this->d'}}
  PointerToMemberDataTest1() {}
};

void fPointerToMemberDataTest1() {
  PointerToMemberDataTest1(); // expected-warning{{1 uninitialized field}}
}

struct PointerToMemberDataTest2 {
  int UsefulFunctions::*d;
  PointerToMemberDataTest2(int UsefulFunctions::*d) : d(d) {
    // All good!
  }
};

void fPointerToMemberDataTest2() {
  int UsefulFunctions::*d = &UsefulFunctions::a;
  PointerToMemberDataTest2 a(d);
}

struct MultiPointerToMemberDataTest1 {
  int UsefulFunctions::**d; // expected-note{{uninitialized pointer 'this->d'}}
  MultiPointerToMemberDataTest1() {}
};

void fMultiPointerToMemberDataTest1() {
  MultiPointerToMemberDataTest1(); // expected-warning{{1 uninitialized field}}
}

struct MultiPointerToMemberDataTest2 {
  int UsefulFunctions::**d;
  MultiPointerToMemberDataTest2(int UsefulFunctions::**d) : d(d) {
    // All good!
  }
};

void fMultiPointerToMemberDataTest2() {
  int UsefulFunctions::*d = &UsefulFunctions::a;
  MultiPointerToMemberDataTest2 a(&d);
}
#endif // PEDANTIC

//===----------------------------------------------------------------------===//
// Tests for list-like records.
//===----------------------------------------------------------------------===//

class ListTest1 {
public:
  struct Node {
    Node *next = nullptr; // no crash
    int i;
  };

private:
  Node *head = nullptr;

public:
  ListTest1() {
    // All good!
  }
};

void fListTest1() {
  ListTest1();
}

class ListTest2 {
public:
  struct Node {
    Node *next = nullptr;
    int i; // expected-note{{uninitialized field 'this->head->i'}}
  };

private:
  Node *head = nullptr;

public:
  ListTest2(Node *node, int) : head(node) { // expected-warning{{1 uninitialized field}}
  }
};

void fListTest2() {
  ListTest2::Node n;
  ListTest2(&n, int());
}

class CyclicList {
public:
  struct Node {
    Node *next = nullptr;
    int i; // expected-note{{uninitialized field 'this->head->i'}}
  };

private:
  Node *head = nullptr;

public:
  CyclicList(Node *node, int) : head(node) { // expected-warning{{1 uninitialized field}}
  }
};

void fCyclicList() {
  /*
               n3
              /  \
    this -- n1 -- n2
  */

  CyclicList::Node n1;
  CyclicList::Node n2;
  n2.next = &n1;
  n2.i = 50;
  CyclicList::Node n3;
  n3.next = &n2;
  n3.i = 50;
  n1.next = &n3;
  // note that n1.i is uninitialized
  CyclicList(&n1, int());
}

struct RingListTest {
  RingListTest *next; // no-crash
  RingListTest() : next(this) {}
};

void fRingListTest() {
  RingListTest();
}

//===----------------------------------------------------------------------===//
// Tests for classes containing references.
//===----------------------------------------------------------------------===//

class ReferenceTest1 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  RecordType &lref;
  RecordType &&rref;

public:
  ReferenceTest1(RecordType &lref, RecordType &rref) : lref(lref), rref(static_cast<RecordType &&>(rref)) {
    // All good!
  }
};

void fReferenceTest1() {
  ReferenceTest1::RecordType d{33, 34};
  ReferenceTest1(d, d);
}

#ifdef PEDANTIC
class ReferenceTest2 {
public:
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->lref.x'}}
    int y; // expected-note{{uninitialized field 'this->lref.y'}}
  };

private:
  RecordType &lref;
  RecordType &&rref;

public:
  ReferenceTest2(RecordType &lref, RecordType &rref)
      : lref(lref), rref(static_cast<RecordType &&>(rref)) { // expected-warning{{2 uninitialized fields}}
  }
};

void fReferenceTest2() {
  ReferenceTest2::RecordType c;
  ReferenceTest2(c, c);
}
#else
class ReferenceTest2 {
public:
  struct RecordType {
    int x;
    int y;
  };

private:
  RecordType &lref;
  RecordType &&rref;

public:
  ReferenceTest2(RecordType &lref, RecordType &rref)
      : lref(lref), rref(static_cast<RecordType &&>(rref)) {
  }
};

void fReferenceTest2() {
  ReferenceTest2::RecordType c;
  ReferenceTest2(c, c);
}
#endif // PEDANTIC

class ReferenceTest3 {
public:
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->lref.x'}}
    int y; // expected-note{{uninitialized field 'this->lref.y'}}
  };

private:
  RecordType &lref;
  RecordType &&rref;

public:
  ReferenceTest3(RecordType &lref, RecordType &rref)
      : lref(lref), rref(static_cast<RecordType &&>(rref)) { // expected-warning{{2 uninitialized fields}}
  }
};

void fReferenceTest3() {
  ReferenceTest3::RecordType c, d{35, 36};
  ReferenceTest3(c, d);
}

class ReferenceTest4 {
public:
  struct RecordType {
    int x; // expected-note{{uninitialized field 'this->rref.x'}}
    int y; // expected-note{{uninitialized field 'this->rref.y'}}
  };

private:
  RecordType &lref;
  RecordType &&rref;

public:
  ReferenceTest4(RecordType &lref, RecordType &rref)
      : lref(lref), rref(static_cast<RecordType &&>(rref)) { // expected-warning{{2 uninitialized fields}}
  }
};

void fReferenceTest5() {
  ReferenceTest4::RecordType c, d{37, 38};
  ReferenceTest4(d, c);
}

//===----------------------------------------------------------------------===//
// Tests for objects containing multiple references to the same object.
//===----------------------------------------------------------------------===//

struct IntMultipleReferenceToSameObjectTest {
  int *iptr; // expected-note{{uninitialized pointee 'this->iptr'}}
  int &iref; // no-note, pointee of this->iref was already reported

  int dontGetFilteredByNonPedanticMode = 0;

  IntMultipleReferenceToSameObjectTest(int *i) : iptr(i), iref(*i) {} // expected-warning{{1 uninitialized field}}
};

void fIntMultipleReferenceToSameObjectTest() {
  int a;
  IntMultipleReferenceToSameObjectTest Test(&a);
}

struct IntReferenceWrapper1 {
  int &a; // expected-note{{uninitialized pointee 'this->a'}}

  int dontGetFilteredByNonPedanticMode = 0;

  IntReferenceWrapper1(int &a) : a(a) {} // expected-warning{{1 uninitialized field}}
};

struct IntReferenceWrapper2 {
  int &a; // no-note, pointee of this->a was already reported

  int dontGetFilteredByNonPedanticMode = 0;

  IntReferenceWrapper2(int &a) : a(a) {} // no-warning
};

void fMultipleObjectsReferencingTheSameObjectTest() {
  int a;

  IntReferenceWrapper1 T1(a);
  IntReferenceWrapper2 T2(a);
}
