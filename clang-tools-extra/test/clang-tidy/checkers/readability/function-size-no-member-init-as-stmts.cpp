// RUN: %check_clang_tidy %s readability-function-size %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-function-size.LineThreshold: 0, \
// RUN:         readability-function-size.StatementThreshold: 0, \
// RUN:         readability-function-size.BranchThreshold: 0, \
// RUN:         readability-function-size.ParameterThreshold: 5, \
// RUN:         readability-function-size.NestingThreshold: 2, \
// RUN:         readability-function-size.VariableThreshold: 1, \
// RUN:         readability-function-size.CountMemberInitAsStmt: false \
// RUN:     }}'

// Bad formatting is intentional, don't run clang-format over the whole file!

void foo1() {
}

void foo2() {;}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'foo2' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 1 statements (threshold 0)

struct A {
  A(int c, int d) : a(0), b(c) { ; }
  int a;
  int b;
};
// CHECK-MESSAGES: :[[@LINE-4]]:3: warning: function 'A' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: 1 statements (threshold 0)

struct B {
  B(int x, int y, int z) : a(x + y * z), b(), c_a(y, z) {
    ;
  }
  int a;
  int b;
  A c_a;
};
// CHECK-MESSAGES: :[[@LINE-7]]:3: warning: function 'B' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-8]]:3: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: 1 statements (threshold 0)

struct C : A, B {
  // 0 statements
  C() : A(0, 4), B(1, 2, 3) {}
};

template<typename T>
struct TemplateC {
  // 0 statements
  TemplateC() : a(3) {}
  T a;
};

template<typename T>
struct TemplateD {
  template<typename U>
  TemplateD(U&& val) : member(val) { 
    ;
  }
  
  T member;
};
// CHECK-MESSAGES: :[[@LINE-6]]:3: warning: function 'TemplateD<T>' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-7]]:3: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-8]]:3: note: 1 statements (threshold 0)

void instantiate() {
  TemplateC<int> c;
  TemplateD<int> d(5);
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'instantiate' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 2 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 2 variables (threshold 1)
