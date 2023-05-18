// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

void clang_analyzer_eval(bool);

void array_init() {
  int arr[] = {1, 2, 3, 4, 5};

  auto [a, b, c, d, e] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(e == 5); // expected-warning{{TRUE}}
}

void array_uninit() {
  int arr[5];

  auto [a, b, c, d, e] = arr;

  int x = e; // expected-warning{{Assigned value is garbage or undefined}}
}

void lambda_init() {
  int arr[] = {1, 2, 3, 4, 5};

  auto l = [arr] { return arr[0]; }();
  clang_analyzer_eval(l == 1); // expected-warning{{TRUE}}

  l = [arr] { return arr[1]; }();
  clang_analyzer_eval(l == 2); // expected-warning{{TRUE}}

  l = [arr] { return arr[2]; }();
  clang_analyzer_eval(l == 3); // expected-warning{{TRUE}}

  l = [arr] { return arr[3]; }();
  clang_analyzer_eval(l == 4); // expected-warning{{TRUE}}

  l = [arr] { return arr[4]; }();
  clang_analyzer_eval(l == 5); // expected-warning{{TRUE}}
}

void lambda_uninit() {
  int arr[5];

  // FIXME: These should be Undefined, but we fail to read Undefined from a lazyCompoundVal
  int l = [arr] { return arr[0]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[1]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[2]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[3]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[4]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}
}

struct S {
  int arr[5];
};

void copy_ctor_init() {
  S orig;
  orig.arr[0] = 1;
  orig.arr[1] = 2;
  orig.arr[2] = 3;
  orig.arr[3] = 4;
  orig.arr[4] = 5;

  S copy = orig;
  clang_analyzer_eval(copy.arr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[4] == 5); // expected-warning{{TRUE}}
}

void copy_ctor_uninit() {
  S orig;

  S copy = orig;

  // FIXME: These should be Undefined, but we fail to read Undefined from a lazyCompoundVal.
  // If the struct is not considered a small struct, instead of a copy, we store a lazy compound value.
  // As the struct has an array data member, it is not considered small.
  clang_analyzer_eval(copy.arr[0]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[1]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[3]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[4]); // expected-warning{{UNKNOWN}}
}

void move_ctor_init() {
  S orig;
  orig.arr[0] = 1;
  orig.arr[1] = 2;
  orig.arr[2] = 3;
  orig.arr[3] = 4;
  orig.arr[4] = 5;

  S moved = (S &&) orig;

  clang_analyzer_eval(moved.arr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[4] == 5); // expected-warning{{TRUE}}
}

void move_ctor_uninit() {
  S orig;

  S moved = (S &&) orig;

  // FIXME: These should be Undefined, but we fail to read Undefined from a lazyCompoundVal.
  clang_analyzer_eval(moved.arr[0]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[1]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[3]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[4]); // expected-warning{{UNKNOWN}}
}

// The struct has a user defined copy and move ctor, which allow us to
// track the values more precisely when an array of this struct is being
// copy/move initialized by ArrayInitLoopExpr.
struct S2 {
  inline static int c = 0;
  int i;

  S2() : i(++c) {}

  S2(const S2 &copy) {
    i = copy.i + 1;
  }

  S2(S2 &&move) {
    i = move.i + 2;
  }
};

void array_init_non_pod() {
  S2::c = 0;
  S2 arr[4];

  auto [a, b, c, d] = arr;

  clang_analyzer_eval(a.i == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(c.i == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(d.i == 5); // expected-warning{{TRUE}}
}

struct S3 {
  int i;
};

// The duplicate is required to emit a warning at 2 different places. 
struct S3_duplicate {
  int i;
};

void array_uninit_non_pod() {
  S3 arr[1];

  auto [a] = arr; // expected-warning@159{{ in implicit constructor is garbage or undefined }}
}

void lambda_init_non_pod() {
  S2::c = 0;
  S2 arr[4];

  auto l = [arr] { return arr[0].i; }();
  clang_analyzer_eval(l == 2); // expected-warning{{TRUE}}

  l = [arr] { return arr[1].i; }();
  clang_analyzer_eval(l == 3); // expected-warning{{TRUE}}

  l = [arr] { return arr[2].i; }();
  clang_analyzer_eval(l == 4); // expected-warning{{TRUE}}

  l = [arr] { return arr[3].i; }();
  clang_analyzer_eval(l == 5); // expected-warning{{TRUE}}
}

void lambda_uninit_non_pod() {
  S3_duplicate arr[4];

  int l = [arr] { return arr[3].i; }(); // expected-warning@164{{ in implicit constructor is garbage or undefined }}
}

// If this struct is being copy/move constructed by the implicit ctors, ArrayInitLoopExpr
// is responsible for the initialization of 'arr' by copy/move constructing each of the
// elements.
struct S5 {
  S2 arr[4];
};

void copy_ctor_init_non_pod() {
  S2::c = 0;
  S5 orig;

  S5 copy = orig;
  clang_analyzer_eval(copy.arr[0].i == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[1].i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[2].i == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[3].i == 5); // expected-warning{{TRUE}}
}

void move_ctor_init_non_pod() {
  S2::c = 0;
  S5 orig;

  S5 moved = (S5 &&) orig;

  clang_analyzer_eval(moved.arr[0].i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[1].i == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[2].i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[3].i == 6); // expected-warning{{TRUE}}
}

//Note: This is the only solution I could find to check the values without 
// crashing clang. For more details on the crash see Issue #57135.
void lambda_capture_multi_array() {
  S3 arr[2][2] = {1,2,3,4};

  {
    int x = [arr] { return arr[0][0].i; }();
    clang_analyzer_eval(x == 1); // expected-warning{{TRUE}}
  }

  {
    int x = [arr] { return arr[0][1].i; }();
    clang_analyzer_eval(x == 2); // expected-warning{{TRUE}}
  }

  {
    int x = [arr] { return arr[1][0].i; }();
    clang_analyzer_eval(x == 3); // expected-warning{{TRUE}}
  }

  {
    int x = [arr] { return arr[1][1].i; }();
    clang_analyzer_eval(x == 4); // expected-warning{{TRUE}}
  }
}

// This struct will force constructor inlining in MultiWrapper.
struct UserDefinedCtor {
  int i;
  UserDefinedCtor() {}
  UserDefinedCtor(const UserDefinedCtor &copy) {
    int j = 1;
    i = copy.i;
  }
};

struct MultiWrapper {
  UserDefinedCtor arr[2][2];
};

void copy_ctor_multi() {
  MultiWrapper MW;

  MW.arr[0][0].i = 0;
  MW.arr[0][1].i = 1;
  MW.arr[1][0].i = 2;
  MW.arr[1][1].i = 3;

  MultiWrapper MWCopy = MW;
  
  clang_analyzer_eval(MWCopy.arr[0][0].i == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(MWCopy.arr[0][1].i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(MWCopy.arr[1][0].i == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(MWCopy.arr[1][1].i == 3); // expected-warning{{TRUE}}
} 

void move_ctor_multi() {
  MultiWrapper MW;

  MW.arr[0][0].i = 0;
  MW.arr[0][1].i = 1;
  MW.arr[1][0].i = 2;
  MW.arr[1][1].i = 3;

  MultiWrapper MWMove = (MultiWrapper &&) MW;
  
  clang_analyzer_eval(MWMove.arr[0][0].i == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(MWMove.arr[0][1].i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(MWMove.arr[1][0].i == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(MWMove.arr[1][1].i == 3); // expected-warning{{TRUE}}
} 

void structured_binding_multi() {
  S3 arr[2][2] = {1,2,3,4};

  auto [a,b] = arr;

  clang_analyzer_eval(a[0].i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a[1].i == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[0].i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[1].i == 4); // expected-warning{{TRUE}}
}

// This snippet used to crash
namespace crash {

struct S
{
  int x;
  S() { x = 1; }
};

void no_crash() {
  S arr[0];
  int n = 1;

  auto l = [arr, n] { return n; };

  int x = l();
  clang_analyzer_eval(x == 1); // expected-warning{{TRUE}}

  // FIXME: This should be 'Undefined'.
  clang_analyzer_eval(arr[0].x); // expected-warning{{UNKNOWN}}
}

} // namespace crash
