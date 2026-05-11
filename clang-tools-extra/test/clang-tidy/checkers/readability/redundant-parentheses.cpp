// RUN: %check_clang_tidy %s readability-redundant-parentheses %t

void parenExpr() {
  1 + 1;
  (1 + 1);
  ((1 + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (1 + 1);
  (((1 + 1)));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-MESSAGES: :[[@LINE-2]]:4: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (1 + 1);
  ((((1 + 1))));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-MESSAGES: :[[@LINE-2]]:4: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (1 + 1);
}

#define EXP (1 + 1)
#define PAREN(e) (e)
void parenExprWithMacro() {
  EXP; // 1
  (EXP); // 2
  ((EXP)); // 3
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    (EXP); // 3
  PAREN((1));
}

void constant() {
  (1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    1;
  (1.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    1.0;
  (true);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    true;
  (',');
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    ',';
  ("v4");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    "v4";
  (nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    nullptr;
}

void declRefExpr(int a) {
  (a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    a;
}

void exceptions() {
  sizeof(1);
  alignof(2);
  alignof((3));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES:    alignof(3);
}

namespace std {
  template<class T> T max(T, T);
  template<class T> T min(T, T);
} // namespace std
void ignoreStdMaxMin() {
  (std::max)(1,2);
  (std::min)(1,2);
}

struct Foo
{
  bool x;
  struct Y {
    bool z;
  } y;
  void foo()
  {
   if ((x)) {
     // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant parentheses around expression [readability-redundant-parentheses]
     // CHECK-FIXES:    if (x) {
   }
   if((this->x)) {
     // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant parentheses around expression [readability-redundant-parentheses]
     // CHECK-FIXES:    if(this->x) {
   }
  }
  bool bar() {
    return true;
  }

  Y fooBar() {
    Y y{};
    return y;
  }
};

void memberExpr() {
  Foo foo{};
  if ((foo.x)) {
   // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant parentheses around expression [readability-redundant-parentheses]
   // CHECK-FIXES:    if (foo.x) {
  }

  if ((foo.y.z)) {
   // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant parentheses around expression [readability-redundant-parentheses]
   // CHECK-FIXES:    if (foo.y.z) {
  }

  if ((foo.bar())) {
   // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant parentheses around expression [readability-redundant-parentheses]
   // CHECK-FIXES:    if (foo.bar()) {
  }

  if ((foo.fooBar().z)) {
   // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant parentheses around expression [readability-redundant-parentheses]
   // CHECK-FIXES:    if (foo.fooBar().z) {
  }
}

int (x);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant parentheses in declaration
// CHECK-FIXES: int x;

void f(int (arg));
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant parentheses in declaration
// CHECK-FIXES: void f(int arg);

int ((nestedX));
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant parentheses in declaration
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: redundant parentheses in declaration
// CHECK-FIXES: int nestedX;

void nestedParam(int ((arg)));
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant parentheses in declaration
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: redundant parentheses in declaration
// CHECK-FIXES: void nestedParam(int arg);

int (&referenceVar) = x;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant parentheses in declaration
// CHECK-FIXES: int &referenceVar = x;

struct S {};
int (S::*memberPtr);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant parentheses in declaration
// CHECK-FIXES: int S::*memberPtr;

int (arrayVar)[2];
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant parentheses in declaration
// CHECK-FIXES: int arrayVar[2];

template <class T>
void templatedParam(T (arg));
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant parentheses in declaration
// CHECK-FIXES: void templatedParam(T arg);

template <class T>
struct TemplateStruct {
  T (member);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant parentheses in declaration
// CHECK-FIXES: T member;
};

// Negative cases.
int (*functionPtr)(int);
void (*callback)(int);
int (*arrayPtr[2])(int);
#define DECL_WITH_PARENS(name) int (name)
DECL_WITH_PARENS(macroVar);
#define PAREN_NAME(name) (name)
int PAREN_NAME(macroName);
using AliasName = int;
void instantiateTemplates() {
  templatedParam<int>(0);
  TemplateStruct<int> s;
}