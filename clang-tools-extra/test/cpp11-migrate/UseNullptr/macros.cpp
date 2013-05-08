// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-nullptr %t.cpp -- -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t2.cpp
// RUN: cpp11-migrate -use-nullptr -user-null-macros=MY_NULL %t2.cpp -- -I %S
// RUN: FileCheck -check-prefix=USER-SUPPLIED-NULL -input-file=%t2.cpp %s

#define NULL 0
// CHECK: #define NULL 0

void dummy(int*) {}
void side_effect() {}

#define MACRO_EXPANSION_HAS_NULL \
  void foo() { \
    dummy(0); \
    dummy(NULL); \
    side_effect(); \
  }
  // CHECK: void foo() { \
  // CHECK-NEXT:   dummy(0); \
  // CHECK-NEXT:   dummy(NULL); \

MACRO_EXPANSION_HAS_NULL;
// CHECK: MACRO_EXPANSION_HAS_NULL;
#undef MACRO_EXPANSION_HAS_NULL


void test_macro_expansion1() {
#define MACRO_EXPANSION_HAS_NULL \
  dummy(NULL); \
  side_effect();
  // CHECK: dummy(NULL); \
  // CHECK-NEXT: side_effect();

  MACRO_EXPANSION_HAS_NULL;
  // CHECK: MACRO_EXPANSION_HAS_NULL;

#undef MACRO_EXPANSION_HAS_NULL
}

// Test macro expansion with cast sequence, PR15572
void test_macro_expansion2() {
#define MACRO_EXPANSION_HAS_NULL \
  dummy((int*)0); \
  side_effect();
  // CHECK: dummy((int*)0); \
  // CHECK-NEXT: side_effect();

  MACRO_EXPANSION_HAS_NULL;
  // CHECK: MACRO_EXPANSION_HAS_NULL;

#undef MACRO_EXPANSION_HAS_NULL
}

void test_macro_expansion3() {
#define MACRO_EXPANSION_HAS_NULL \
  dummy(NULL); \
  side_effect();
  // CHECK: dummy(NULL); \
  // CHECK-NEXT: side_effect();

#define OUTER_MACRO \
  MACRO_EXPANSION_HAS_NULL; \
  side_effect();

  OUTER_MACRO;
  // CHECK: OUTER_MACRO;

#undef OUTER_MACRO
#undef MACRO_EXPANSION_HAS_NULL
}

void test_macro_expansion4() {
#define MY_NULL NULL
  int *p = MY_NULL;
  // CHECK: int *p = MY_NULL;
  // USER-SUPPLIED-NULL: int *p = nullptr;
#undef MY_NULL
}

#define IS_EQ(x, y) if (x != y) return;
void test_macro_args() {
  int i = 0;
  int *Ptr;

  IS_EQ(static_cast<int*>(0), Ptr);
  // CHECK: IS_EQ(static_cast<int*>(nullptr), Ptr);
  IS_EQ(0, Ptr);    // literal
  // CHECK: IS_EQ(nullptr, Ptr);
  IS_EQ(NULL, Ptr); // macro
  // CHECK: IS_EQ(nullptr, Ptr);

  // These are ok since the null literal is not spelled within a macro.
#define myassert(x) if (!(x)) return;
  myassert(0 == Ptr);
  // CHECK: myassert(nullptr == Ptr);
  myassert(NULL == Ptr);
  // CHECK: myassert(nullptr == Ptr);

  // These are bad as the null literal is buried in a macro.
#define BLAH(X) myassert(0 == (X));
  // CHECK: #define BLAH(X) myassert(0 == (X));
#define BLAH2(X) myassert(NULL == (X));
  // CHECK: #define BLAH2(X) myassert(NULL == (X));
  BLAH(Ptr);
  // CHECK: BLAH(Ptr);
  BLAH2(Ptr);
  // CHECK: BLAH2(Ptr);

  // Same as above but testing extra macro expansion.
#define EXPECT_NULL(X) IS_EQ(0, X);
  // CHECK: #define EXPECT_NULL(X) IS_EQ(0, X);
#define EXPECT_NULL2(X) IS_EQ(NULL, X);
  // CHECK: #define EXPECT_NULL2(X) IS_EQ(NULL, X);
  EXPECT_NULL(Ptr);
  // CHECK: EXPECT_NULL(Ptr);
  EXPECT_NULL2(Ptr);
  // CHECK: EXPECT_NULL2(Ptr);

  // Almost the same as above but now null literal is not in a macro so ok
  // to transform.
#define EQUALS_PTR(X) IS_EQ(X, Ptr);
  EQUALS_PTR(0);
  EQUALS_PTR(NULL);

  // Same as above but testing extra macro expansion.
#define EQUALS_PTR_I(X) EQUALS_PTR(X)
  EQUALS_PTR_I(0);
  // CHECK: EQUALS_PTR_I(nullptr);
  EQUALS_PTR_I(NULL);
  // CHECK: EQUALS_PTR_I(nullptr);

  // Ok since null literal not within macro. However, now testing macro
  // used as arg to another macro.
#define decorate(EXPR) side_effect(); EXPR;
  decorate(IS_EQ(NULL, Ptr));
  // CHECK: decorate(IS_EQ(nullptr, Ptr));
  decorate(IS_EQ(0, Ptr));
  // CHECK: decorate(IS_EQ(nullptr, Ptr));

  // This macro causes a NullToPointer cast to happen where 0 is assigned to z
  // but the 0 literal cannot be replaced because it is also used as an
  // integer in the comparison.
#define INT_AND_PTR_USE(X) do { int *z = X; if (X == 4) break; } while(false)
  INT_AND_PTR_USE(0);
  // CHECK: INT_AND_PTR_USE(0);

  // Both uses of X in this case result in NullToPointer casts so replacement
  // is possible.
#define PTR_AND_PTR_USE(X) do { int *z = X; if (X != z) break; } while(false)
  PTR_AND_PTR_USE(0);
  // CHECK: PTR_AND_PTR_USE(nullptr);
  PTR_AND_PTR_USE(NULL);
  // CHECK: PTR_AND_PTR_USE(nullptr);
}
