// RUN: %check_clang_tidy %s readability-function-size %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-function-size.LineThreshold: 0, \
// RUN:         readability-function-size.StatementThreshold: 0, \
// RUN:         readability-function-size.BranchThreshold: 0, \
// RUN:         readability-function-size.ParameterThreshold: 5, \
// RUN:         readability-function-size.NestingThreshold: 2, \
// RUN:         readability-function-size.VariableThreshold: 1 \
// RUN:     }}'


// RUN: %check_clang_tidy -check-suffixes=OPTIONAL %s readability-function-size %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-function-size.StatementThreshold: "-1", \
// RUN:         readability-function-size.BranchThreshold: "5", \
// RUN:         readability-function-size.ParameterThreshold: "none", \
// RUN:         readability-function-size.NestingThreshold: "", \
// RUN:         readability-function-size.VariableThreshold: "" \
// RUN:     }}'

// Bad formatting is intentional, don't run clang-format over the whole file!

void foo1() {
}

void foo2() {;}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'foo2' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 1 statements (threshold 0)

void foo3() {
;

}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'foo3' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 1 statements (threshold 0)

void foo4(int i) { if (i) {} else; {}
}
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: function 'foo4' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 1 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 3 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 branches (threshold 0)

void foo5(int i) {for(;i;)while(i)
do;while(i);
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'foo5' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 7 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 3 branches (threshold 0)

template <typename T> T foo6(T i) {return i;
}
int x = foo6(0);
// CHECK-MESSAGES: :[[@LINE-3]]:25: warning: function 'foo6' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-4]]:25: note: 1 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:25: note: 1 statements (threshold 0)

void foo7(int p1, int p2, int p3, int p4, int p5, int p6) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'foo7' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 1 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 6 parameters (threshold 5)

void bar1() { [](){;;;;;;;;;;;if(1){}}();


}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'bar1' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 14 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 1 branches (threshold 0)

void bar2() { class A { void barx() {;;} }; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'bar2' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 3 statements (threshold 0)
//
// CHECK-MESSAGES: :[[@LINE-4]]:30: warning: function 'barx' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-5]]:30: note: 2 statements (threshold 0)

#define macro() {int x; {int y; {int z;}}}

void baz0() { // 1
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'baz0' exceeds recommended size/complexity
  // CHECK-MESSAGES: :[[@LINE-2]]:6: note: 28 lines including whitespace and comments (threshold 0)
  // CHECK-MESSAGES: :[[@LINE-3]]:6: note: 9 statements (threshold 0)
  int a;
  { // 2
    int b;
    { // 3
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: nesting level 3 starts here (threshold 2)
      int c;
      { // 4
        int d;
      }
    }
  }
  { // 2
    int e;
  }
  { // 2
    { // 3
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: nesting level 3 starts here (threshold 2)
      int j;
    }
  }
  macro()
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: nesting level 3 starts here (threshold 2)
  // CHECK-MESSAGES: :[[@LINE-28]]:25: note: expanded from macro 'macro'
  // CHECK-MESSAGES: :[[@LINE-27]]:6: note: 9 variables (threshold 1)
}

// check that nested if's are not reported. this was broken initially
void nesting_if() { // 1
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'nesting_if' exceeds recommended size/complexity
  // CHECK-MESSAGES: :[[@LINE-2]]:6: note: 25 lines including whitespace and comments (threshold 0)
  // CHECK-MESSAGES: :[[@LINE-3]]:6: note: 18 statements (threshold 0)
  // CHECK-MESSAGES: :[[@LINE-4]]:6: note: 6 branches (threshold 0)
  // CHECK-MESSAGES-OPTIONAL: :[[@LINE-5]]:6: warning: function 'nesting_if' exceeds recommended size/complexity
  // CHECK-MESSAGES-OPTIONAL: :[[@LINE-6]]:6: note: 6 branches (threshold 5)
  if (true) { // 2
     int j;
  } else if (true) { // 2
     int j;
     if (true) { // 3
       // CHECK-MESSAGES: :[[@LINE-1]]:16: note: nesting level 3 starts here (threshold 2)
       int j;
     }
  } else if (true) { // 2
     int j;
     if (true) { // 3
       // CHECK-MESSAGES: :[[@LINE-1]]:16: note: nesting level 3 starts here (threshold 2)
       int j;
     }
  } else if (true) { // 2
     int j;
  }
  // CHECK-MESSAGES: :[[@LINE-24]]:6: note: 6 variables (threshold 1)
}

// however this should warn
void bad_if_nesting() { // 1
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'bad_if_nesting' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 23 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 12 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 4 branches (threshold 0)
  if (true) {    // 2
    int j;
  } else { // 2
    if (true) { // 3
      // CHECK-MESSAGES: :[[@LINE-1]]:15: note: nesting level 3 starts here (threshold 2)
      int j;
    } else { // 3
      // CHECK-MESSAGES: :[[@LINE-1]]:12: note: nesting level 3 starts here (threshold 2)
      if (true) { // 4
        int j;
      } else { // 4
        if (true) { // 5
          int j;
        }
      }
    }
  }
  // CHECK-MESSAGES: :[[@LINE-22]]:6: note: 4 variables (threshold 1)
}

void variables_0() {
  int i;
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'variables_0' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 statements (threshold 0)
void variables_1(int i) {
  int j;
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'variables_1' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 statements (threshold 0)
void variables_2(int i, int j) {
  ;
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'variables_2' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 statements (threshold 0)
void variables_3() {
  int i[2];
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'variables_3' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 statements (threshold 0)
void variables_4() {
  int i;
  int j;
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'variables_4' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 2 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 2 variables (threshold 1)
void variables_5() {
  int i, j;
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'variables_5' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 2 variables (threshold 1)
void variables_6() {
  for (int i;;)
    for (int j;;)
      ;
}
// CHECK-MESSAGES: :[[@LINE-5]]:6: warning: function 'variables_6' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 4 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 5 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 2 branches (threshold 0)
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: 2 variables (threshold 1)
void variables_7() {
  if (int a = 1)
    if (int b = 2)
      ;
}
// CHECK-MESSAGES: :[[@LINE-5]]:6: warning: function 'variables_7' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 4 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 7 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 2 branches (threshold 0)
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: 2 variables (threshold 1)
void variables_8() {
  int a[2];
  for (auto i : a)
    for (auto j : a)
      ;
}
// CHECK-MESSAGES: :[[@LINE-6]]:6: warning: function 'variables_8' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 5 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 8 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: 2 branches (threshold 0)
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: 3 variables (threshold 1)
void variables_9() {
  int a, b;
  struct A {
    A(int c, int d) {
      int e, f;
    }
  };
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: warning: function 'variables_9' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: 7 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: 3 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-11]]:6: note: 2 variables (threshold 1)
// CHECK-MESSAGES: :[[@LINE-9]]:5: warning: function 'A' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-10]]:5: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-11]]:5: note: 1 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-12]]:5: note: 2 variables (threshold 1)
void variables_10() {
  int a, b;
  struct A {
    int c;
    int d;
  };
}
// CHECK-MESSAGES: :[[@LINE-7]]:6: warning: function 'variables_10' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 6 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: 2 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: 2 variables (threshold 1)
void variables_11() {
  struct S {
    void bar() {
      int a, b;
    }
  };
}
// CHECK-MESSAGES: :[[@LINE-7]]:6: warning: function 'variables_11' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 6 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:10: warning: function 'bar' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-8]]:10: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-9]]:10: note: 2 variables (threshold 1)
void variables_12() {
  int v;
  auto test = [](int a, int b) -> void {};
  test({}, {});
}
// CHECK-MESSAGES: :[[@LINE-5]]:6: warning: function 'variables_12' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 4 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 3 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 2 variables (threshold 1)
void variables_13() {
  int v;
  auto test = []() -> void {
    int a;
    int b;
  };
  test();
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: warning: function 'variables_13' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: 7 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: 5 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-11]]:6: note: 2 variables (threshold 1)
void variables_14() {
  (void)({int a = 12; a; });
  (void)({int a = 12; a; });
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'variables_14' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 6 statements (threshold 0)
#define SWAP(x, y) ({__typeof__(x) temp = x; x = y; y = temp; })
void variables_15() {
  int a = 10, b = 12;
  SWAP(a, b);
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'variables_15' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 5 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 2 variables (threshold 1)
#define vardecl(type, name) type name;
void variables_16() {
  vardecl(int, a);
  vardecl(int, b);
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'variables_16' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 4 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 2 variables (threshold 1)
