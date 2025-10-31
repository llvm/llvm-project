// RUN: %check_clang_tidy %s readability-isolate-declaration %t -- -- -fexceptions

void f() {
  int i;
}

void f2() {
  int i, j, *k, lala = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int i;
  // CHECK-FIXES: int j;
  // CHECK-FIXES: int *k;
  // CHECK-FIXES: int lala = 42;

  int normal, weird = /* comment */ 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int normal;
  // CHECK-FIXES: int weird = /* comment */ 42;

  int /* here is a comment */ v1,
      // another comment
      v2 = 42 // Ok, more comments
      ;
  // CHECK-MESSAGES: [[@LINE-4]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int /* here is a comment */ v1;
  // CHECK-FIXES: int /* here is a comment */ // another comment
  // CHECK-FIXES:     v2 = 42 // Ok, more comments
  // CHECK-FIXES:     ;

  auto int1 = 42, int2 = 0, int3 = 43;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: auto int1 = 42;
  // CHECK-FIXES: auto int2 = 0;
  // CHECK-FIXES: auto int3 = 43;

  decltype(auto) ptr1 = &int1, ptr2 = &int1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: decltype(auto) ptr1 = &int1;
  // CHECK-FIXES: decltype(auto) ptr2 = &int1;

  decltype(k) ptr3 = &int1, ptr4 = &int1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: decltype(k) ptr3 = &int1;
  // CHECK-FIXES: decltype(k) ptr4 = &int1;
}

void f3() {
  int i, *pointer1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int i;
  // CHECK-FIXES: int *pointer1;
  //
  int *pointer2 = nullptr, *pointer3 = &i;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int *pointer2 = nullptr;
  // CHECK-FIXES: int *pointer3 = &i;

  int *(i_ptr) = nullptr, *((i_ptr2));
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int *(i_ptr) = nullptr;
  // CHECK-FIXES: int *((i_ptr2));

  float(*f_ptr)[42], (((f_value))) = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: float (*f_ptr)[42];
  // CHECK-FIXES: float (((f_value))) = 42;

  float(((*f_ptr2)))[42], ((*f_ptr3)), f_value2 = 42.f;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: float (((*f_ptr2)))[42];
  // CHECK-FIXES: float ((*f_ptr3));
  // CHECK-FIXES: float f_value2 = 42.f;

  float(((*f_ptr4)))[42], *f_ptr5, ((f_value3));
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: float (((*f_ptr4)))[42];
  // CHECK-FIXES: float *f_ptr5;
  // CHECK-FIXES: float ((f_value3));

  void(((*f2))(int)), (*g2)(int, float);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: void (((*f2))(int));
  // CHECK-FIXES: void (*g2)(int, float);

  float(*(*(*f_ptr6)))[42], (*f_ptr7);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: float (*(*(*f_ptr6)))[42];
  // CHECK-FIXES: float (*f_ptr7);
}

void f4() {
  double d = 42. /* foo */, z = 43., /* hi */ y, c /* */ /*  */, l = 2.;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: double d = 42. /* foo */;
  // CHECK-FIXES: double z = 43.;
  // CHECK-FIXES: double /* hi */ y;
  // CHECK-FIXES: double c /* */ /*  */;
  // CHECK-FIXES: double l = 2.;
}

struct SomeClass {
  SomeClass() = default;
  SomeClass(int value);
};

class Point {
  double x;
  double y;

public:
  Point(double x, double y) : x(x), y(y) {}
};

class Rectangle {
  Point TopLeft;
  Point BottomRight;

public:
  Rectangle(Point TopLeft, Point BottomRight) : TopLeft(TopLeft), BottomRight(BottomRight) {}
};

void f5() {
  SomeClass v1, v2(42), v3{42}, v4(42.5);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: SomeClass v1;
  // CHECK-FIXES: SomeClass v2(42);
  // CHECK-FIXES: SomeClass v3{42};
  // CHECK-FIXES: SomeClass v4(42.5);

  SomeClass v5 = 42, *p1 = nullptr;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: SomeClass v5 = 42;
  // CHECK-FIXES: SomeClass *p1 = nullptr;

  Point P1(0., 2.), P2{2., 0.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: Point P1(0., 2.);
  // CHECK-FIXES: Point P2{2., 0.};

  Rectangle R1({0., 0.}, {1., -2.}), R2{{0., 1.}, {1., 0.}}, R3(P1, P2), R4{P1, P2};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: Rectangle R1({0., 0.}, {1., -2.});
  // CHECK-FIXES: Rectangle R2{{[{][{]}}0., 1.}, {1., 0.{{[}][}]}};
  // CHECK-FIXES: Rectangle R3(P1, P2);
  // CHECK-FIXES: Rectangle R4{P1, P2};
}

void f6() {
  int array1[] = {1, 2, 3, 4}, array2[] = {1, 2, 3}, value1, value2 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int array1[] = {1, 2, 3, 4};
  // CHECK-FIXES: int array2[] = {1, 2, 3};
  // CHECK-FIXES: int value1;
  // CHECK-FIXES: int value2 = 42;
}

template <typename T>
struct TemplatedType {
  TemplatedType() = default;
  TemplatedType(T value);
};

void f7() {
  TemplatedType<int> TT1(42), TT2{42}, TT3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: TemplatedType<int> TT1(42);
  // CHECK-FIXES: TemplatedType<int> TT2{42};
  // CHECK-FIXES: TemplatedType<int> TT3;
  //
  TemplatedType<int *> *TT4(nullptr), TT5, **TT6 = nullptr, *const *const TT7{nullptr};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: TemplatedType<int *> *TT4(nullptr);
  // CHECK-FIXES: TemplatedType<int *> TT5;
  // CHECK-FIXES: TemplatedType<int *> **TT6 = nullptr;
  // CHECK-FIXES: TemplatedType<int *> *const *const TT7{nullptr};

  TemplatedType<int &> **TT8(nullptr), *TT9;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: TemplatedType<int &> **TT8(nullptr);
  // CHECK-FIXES: TemplatedType<int &> *TT9;

  TemplatedType<int *> TT10{nullptr}, *TT11(nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: TemplatedType<int *> TT10{nullptr};
  // CHECK-FIXES: TemplatedType<int *> *TT11(nullptr);
}

void forbidden_transformations() {
  for (int i = 0, j = 42; i < j; ++i)
    ;
}

#define NULL 0
#define MY_NICE_TYPE int **
#define VAR_NAME(name) name##__LINE__
#define A_BUNCH_OF_VARIABLES int m1 = 42, m2 = 43, m3 = 44;

void macros() {
  int *p1 = NULL, *p2 = NULL;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int *p1 = NULL;
  // CHECK-FIXES: int *p2 = NULL;

  // Macros are involved, so there will be no transformation
  MY_NICE_TYPE p3, v1, v2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability

  int VAR_NAME(v3),
      VAR_NAME(v4),
      VAR_NAME(v5);
  // CHECK-MESSAGES: [[@LINE-3]]:3: warning: multiple declarations in a single statement reduces readability

  A_BUNCH_OF_VARIABLES
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability

  int Unconditional,
  // Explanatory comment.
#if CONFIGURATION
      IfConfigured = 42,
#else
      IfConfigured = 0;
#endif
  // CHECK-MESSAGES: [[@LINE-7]]:3: warning: multiple declarations in a single statement reduces readability
}

void dontTouchParameter(int param1, int param2) {}

struct StructOne {
  StructOne() {}
  StructOne(int b) {}

  int member1, member2;
  // TODO: Handle FieldDecl's as well
};

using PointerType = int;

struct {
  int i;
} AS1, AS2;
struct TemT {
  template <typename T>
  T *getAs() {
    return nullptr;
  }
} TT1, TT2;

void complex_typedefs() {
  typedef int *IntPtr;
  typedef int ArrayType[2];
  typedef int FunType(void);

  IntPtr intptr1, intptr2 = nullptr, intptr3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: IntPtr intptr1;
  // CHECK-FIXES: IntPtr intptr2 = nullptr;
  // CHECK-FIXES: IntPtr intptr3;

  IntPtr *DoublePtr1 = nullptr, **TriplePtr, SinglePtr = nullptr;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: IntPtr *DoublePtr1 = nullptr;
  // CHECK-FIXES: IntPtr **TriplePtr;
  // CHECK-FIXES: IntPtr SinglePtr = nullptr;

  IntPtr intptr_array1[2], intptr_array2[4] = {nullptr, nullptr, nullptr, nullptr};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: IntPtr intptr_array1[2];
  // CHECK-FIXES: IntPtr intptr_array2[4] = {nullptr, nullptr, nullptr, nullptr};

  ArrayType arraytype1, arraytype2 = {1}, arraytype3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: ArrayType arraytype1;
  // CHECK-FIXES: ArrayType arraytype2 = {1};
  // CHECK-FIXES: ArrayType arraytype3;

  // Don't touch function declarations.
  FunType funtype1, funtype2, functype3;

  for (int index1 = 0, index2 = 0;;) {
    int localFor1 = 1, localFor2 = 2;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: int localFor1 = 1;
    // CHECK-FIXES: int localFor2 = 2;
  }

  StructOne s1, s2(23), s3, s4(3), *sptr = new StructOne(2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: StructOne s1;
  // CHECK-FIXES: StructOne s2(23);
  // CHECK-FIXES: StructOne s3;
  // CHECK-FIXES: StructOne s4(3);
  // CHECK-FIXES: StructOne *sptr = new StructOne(2);

  struct StructOne cs1, cs2(42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: struct StructOne cs1;
  // CHECK-FIXES: struct StructOne cs2(42);

  int *ptrArray[3], dummy, **ptrArray2[5], twoDim[2][3], *twoDimPtr[2][3];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int *ptrArray[3];
  // CHECK-FIXES: int dummy;
  // CHECK-FIXES: int **ptrArray2[5];
  // CHECK-FIXES: int twoDim[2][3];
  // CHECK-FIXES: int *twoDimPtr[2][3];

  {
    void f1(int), g1(int, float);
  }

  {
    void gg(int, float);

    void (*f2)(int), (*g2)(int, float) = gg;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: void (*f2)(int);
    // CHECK-FIXES: void (*g2)(int, float) = gg;

    void /*(*/ (/*(*/ *f3)(int), (*g3)(int, float);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: void /*(*/ (/*(*/ *f3)(int);
    // CHECK-FIXES: void /*(*/ (*g3)(int, float);
  }

  // clang-format off
  auto returner = []() { return int(32); };
  int intfunction = returner(), intarray[] =
          {
                  1,
                  2,
                  3,
                  4
          }, bb = 4;
  // CHECK-MESSAGES: [[@LINE-7]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int intfunction = returner();
  // CHECK-FIXES: int intarray[] =
  // CHECK-FIXES:         {
  // CHECK-FIXES:                 1,
  // CHECK-FIXES:                 2,
  // CHECK-FIXES:                 3,
  // CHECK-FIXES:                 4
  // CHECK-FIXES:         };
  // CHECK-FIXES: int bb = 4;
  // clang-format on

  TemT *T1 = &TT1, *T2 = &TT2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: TemT *T1 = &TT1;
  // CHECK-FIXES: TemT *T2 = &TT2;

  const PointerType *PT1 = T1->getAs<PointerType>(),
                    *PT2 = T2->getAs<PointerType>();
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: const PointerType *PT1 = T1->getAs<PointerType>();
  // CHECK-FIXES: const PointerType *PT2 = T2->getAs<PointerType>();

  const int *p1 = nullptr;
  const int *p2 = nullptr;

  const int *&pref1 = p1, *&pref2 = p2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: const int *&pref1 = p1;
  // CHECK-FIXES: const int *&pref2 = p2;

  // clang-format off
  const char *literal1 = "clang"   "test"\
                         "one",
             *literal2 = "empty", literal3[] = "three";
  // CHECK-MESSAGES: [[@LINE-3]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: const char *literal1 = "clang"   "test"\
  // CHECK-FIXES:                        "one";
  // CHECK-FIXES: const char *literal2 = "empty";
  // CHECK-FIXES: const char literal3[] = "three";
  // clang-format on
}

void g() try {
  int i, j;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int i;
  // CHECK-FIXES: int j;
} catch (...) {
}

struct S {
  int a;
  const int b;
  void f() {}
};

void memberPointers() {
  typedef const int S::*MemPtr;
  MemPtr aaa = &S::a, bbb = &S::b;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: MemPtr aaa = &S::a;
  // CHECK-FIXES: MemPtr bbb = &S::b;
}

typedef int *tptr, tbt;
typedef int (&tfp)(int, long), tarr[10];
typedef int tarr2[10], tct;

template <typename A, typename B>
void should_not_be_touched(A, B);

int variable, function(void);

int call_func_with_sideeffect();
void bad_if_decl() {
  if (true)
    int i, j, k = call_func_with_sideeffect();
}
