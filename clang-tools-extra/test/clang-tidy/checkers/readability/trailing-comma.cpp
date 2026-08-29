// RUN: %check_clang_tidy -std=c++98-or-later %s readability-trailing-comma %t

struct S { int x, y; };

enum E1 {
  A,
  B,
  C
};
// CHECK-MESSAGES: :[[@LINE-2]]:4: warning: enum should have a trailing comma [readability-trailing-comma]
// CHECK-FIXES: enum E1 {
// CHECK-FIXES-NEXT:   A,
// CHECK-FIXES-NEXT:   B,
// CHECK-FIXES-NEXT:   C,
// CHECK-FIXES-NEXT: };

enum E2 {
  V = 1
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: enum should have a trailing comma
// CHECK-FIXES: enum E2 {
// CHECK-FIXES-NEXT:   V = 1,
// CHECK-FIXES-NEXT: };

enum SingleLine { A1, B1, C1 };

enum E3 {
  P,
  Q,
};

enum SingleEnum1 { ONE };   
enum SingleEnum2 { TWO, };  
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: enum should not have a trailing comma
// CHECK-FIXES: enum SingleEnum2 { TWO };  
enum SingleEnum3 {
  THREE
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: enum should have a trailing comma
// CHECK-FIXES: enum SingleEnum3 {
// CHECK-FIXES-NEXT:    THREE,
// CHECK-FIXES-NEXT:  };
enum SingleEnum4 {
  FOUR,
};

enum Empty {};

void f() {
  int a[] = {
    1
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should have a trailing comma
  // CHECK-FIXES: int a[] = {
  // CHECK-FIXES-NEXT:     1,
  // CHECK-FIXES-NEXT:   };

  S s = {
    1,
    2
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should have a trailing comma
  // CHECK-FIXES: S s = {
  // CHECK-FIXES-NEXT:     1,
  // CHECK-FIXES-NEXT:     2,
  // CHECK-FIXES-NEXT:   };

  int b[] = {1, 2, 3};
  S s2 = {1, 2};

  int e[] = {1, 2, 3,};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int e[] = {1, 2, 3};

  int c[] = {
    1,
    2,
  };
  int d[] = {};

  int single1[] = {1};
  int single2[] = {1,};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should not have a trailing comma
  // CHECK-FIXES:  int single2[] = {1};       
  int single3[] = {
    1
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should have a trailing comma
  // CHECK-FIXES:  int single3[] = {
  // CHECK-FIXES-NEXT:    1,
  // CHECK-FIXES-NEXT:  };
  int single4[] = {
    1,
  };
  S singleS1 = {42};          
  S singleS2 = {42,};         
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: initializer list should not have a trailing comma
  // CHECK-FIXES:  S singleS2 = {42};      
}

struct N { S a, b; };
void nested() {
  N n = {{1, 2}, {3, 4}};
  N n2 = {{3, 4,},};
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: N n2 = {{[{][{]3, 4[}][}]}};
}

void nestedMultiLine() {
  N n = {
    {1, 2},
    {3, 4}
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: initializer list should have a trailing comma
  // CHECK-FIXES: N n = {
  // CHECK-FIXES-NEXT:     {1, 2},
  // CHECK-FIXES-NEXT:     {3, 4},
  // CHECK-FIXES-NEXT:   };

  N n2 = {
    {1, 2},
    {3, 4},
  };

  struct Container { int arr[3]; };
  Container c1 = {{}};
  Container c2 = {{},};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should not have a trailing comma
  // CHECK-FIXES:   Container c2 = {{[{][{][}][}]}};

  struct Wrapper { S s; };
  Wrapper w1 = {{1}};
  Wrapper w2 = {{1,}};
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: initializer list should not have a trailing comma
  // CHECK-FIXES:   Wrapper w2 = {{[{][{]1[}][}]}};

  Wrapper w3 = {
    {1}
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: initializer list should have a trailing comma
  // CHECK-FIXES: Wrapper w3 = {
  // CHECK-FIXES-NEXT:     {1},
  // CHECK-FIXES-NEXT:   };
}

// Preprocessor directives immediately before '}' must not be treated as the
// last enumerator. Both branches already have trailing commas; a false
// positive would insert a comma after '#endif'.
enum color_t {
  COLOR_RED = 0,
  COLOR_GREEN = 1,
  COLOR_BLUE = 2,
  COLOR_CYAN = 3,
#ifdef USE_MAGENTA
  COLOR_LAST = COLOR_CYAN,
#else
  COLOR_LAST = COLOR_BLUE,
#endif
};

enum GuardedEnumerator {
  GE_A,
  GE_B,
#ifdef USE_EXTRA
  GE_C,
#endif
};

// An enumerator named 'endif' is still diagnosed; only '#endif' is ignored.
enum EndsWithEndifName {
  foo,
  endif
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: enum should have a trailing comma
// CHECK-FIXES: enum EndsWithEndifName {
// CHECK-FIXES-NEXT:   foo,
// CHECK-FIXES-NEXT:   endif,
// CHECK-FIXES-NEXT: };

// Macros are ignored
#define ENUM(n, a, b) enum n { a, b }
#define INIT {1, 2}
#define ITEMS 1,2

ENUM(E1M, Xm, Ym);
int macroArr[] = INIT;
int a[] = { ITEMS };

// Comma from macro should not trigger false positive
#define COMMA ,
int comma_from_macro[] = {
    1
    COMMA
};
