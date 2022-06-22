// RUN: %check_clang_tidy %s readability-delete-null-pointer %t -- -- -fno-delayed-template-parsing

#define NULL 0

template <typename T>
struct Templ {
  void foo() {
    // t1
    if (mem) // t2
      delete mem;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 'if' statement is unnecessary;
    // CHECK-FIXES: // t1
    // CHECK-FIXES-NEXT: {{^    }}// t2
    // CHECK-FIXES-NEXT: delete mem;
  }
  T mem;
};

template <typename T>
struct TemplPtr {
  void foo() {
    // t3
    if (mem) // t4
      delete mem;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 'if' statement is unnecessary;
    // CHECK-FIXES: // t3
    // CHECK-FIXES-NEXT: {{^    }}// t4
    // CHECK-FIXES-NEXT: delete mem;
  }
  T *mem;
};

void instantiate() {
  Templ<int *> ti2;
  ti2.foo();
  TemplPtr<int> ti3;
  ti3.foo();
}

template <typename T>
struct NeverInstantiated {
  void foo() {
    // t1
    if (mem) // t2
      delete mem;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 'if' statement is unnecessary;
    // CHECK-FIXES: // t1
    // CHECK-FIXES-NEXT: {{^    }}// t2
    // CHECK-FIXES-NEXT: delete mem;
  }
  T mem;
};

template <typename T>
struct NeverInstantiatedPtr {
  void foo() {
    // t3
    if (mem) // t4
      delete mem;
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: 'if' statement is unnecessary;
    // CHECK-FIXES: // t3
    // CHECK-FIXES-NEXT: {{^    }}// t4
    // CHECK-FIXES-NEXT: delete mem;
  }
  T *mem;
};

void f() {
  int *ps = 0;
  if (ps /**/) // #0
    delete ps;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: 'if' statement is unnecessary; deleting null pointer has no effect [readability-delete-null-pointer]

  // CHECK-FIXES: int *ps = 0;
  // CHECK-FIXES-NEXT: {{^  }}// #0
  // CHECK-FIXES-NEXT: delete ps;

  int *p = 0;

  // #1
  if (p) { // #2
    delete p;
  } // #3
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: 'if' statement is unnecessary; deleting null pointer has no effect [readability-delete-null-pointer]

  // CHECK-FIXES: {{^  }}// #1
  // CHECK-FIXES-NEXT: {{^  }}// #2
  // CHECK-FIXES-NEXT: delete p;
  // CHECK-FIXES-NEXT: {{^  }}// #3

  int *p2 = new int[3];
  // #4
  if (p2) // #5
    delete[] p2;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: 'if' statement is unnecessary;

  // CHECK-FIXES: // #4
  // CHECK-FIXES-NEXT: {{^  }}// #5
  // CHECK-FIXES-NEXT: delete[] p2;

  int *p3 = 0;
  if (NULL != p3) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'if' statement is unnecessary;
    delete p3;
  }
  // CHECK-FIXES-NOT: if (NULL != p3) {
  // CHECK-FIXES: delete p3;

  int *p4 = nullptr;
  if (p4 != nullptr) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'if' statement is unnecessary;
    delete p4;
  }
  // CHECK-FIXES-NOT: if (p4 != nullptr) {
  // CHECK-FIXES: delete p4;

  char *c;
  if (c != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'if' statement is unnecessary;
    delete c;
  }
  // CHECK-FIXES-NOT: if (c != 0) {
  // CHECK-FIXES: delete c;

  char *c2;
  if (c2) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'if' statement is unnecessary;
    // CHECK-FIXES: } else {
    // CHECK-FIXES: c2 = c;
    delete c2;
  } else {
    c2 = c;
  }
  struct A {
    void foo() {
      if (mp) // #6
        delete mp;
      // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: 'if' statement is unnecessary; deleting null pointer has no effect [readability-delete-null-pointer]
      // CHECK-FIXES: {{^      }}// #6
      // CHECK-FIXES-NEXT: delete mp;
    }
    int *mp;
  };
}

void g() {
  int *p5, *p6;
  if (p5)
    delete p6;

  if (p5 && p6)
    delete p5;

  if (p6) {
    int x = 5;
    delete p6;
  }
}
