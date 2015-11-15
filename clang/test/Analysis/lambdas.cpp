// RUN: %clang_cc1 -std=c++11 -fsyntax-only -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s 
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -analyze -analyzer-checker=core,debug.DumpCFG -analyzer-config inline-lambdas=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

struct X { X(const X&); };
void f(X x) { (void) [x]{}; }


// Lambda semantics tests.

void basicCapture() {
  int i = 5;
  [i]() mutable {
    // clang_analyzer_eval does nothing in inlined functions.
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

void deferredLambdaCall() {
  int i = 5;
  auto l1 = [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  };
  auto l2 = [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  };
  auto l3 = [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  };
  l1();
  l2();
  l3();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

void multipleCaptures() {
  int i = 5, j = 5;
  [i, &j]() mutable {
    if (i != 5 && j != 5)
      clang_analyzer_warnIfReached();
    ++i;
    ++j;
  }();
  clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 6); // expected-warning{{TRUE}}
  [=]() mutable {
    if (i != 5 && j != 6)
      clang_analyzer_warnIfReached();
    ++i;
    ++j;
  }();
  clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 6); // expected-warning{{TRUE}}
  [&]() mutable {
    if (i != 5 && j != 6)
      clang_analyzer_warnIfReached();
    ++i;
    ++j;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 7); // expected-warning{{TRUE}}
}

void testReturnValue() {
  int i = 5;
  auto l = [i] (int a) {
    return i + a;
  };
  int b = l(3);
  clang_analyzer_eval(b == 8); // expected-warning{{TRUE}}
}

void testAliasingBetweenParameterAndCapture() {
  int i = 5;

  auto l = [&i](int &p) {
    i++;
    p++;
  };
  l(i);
  clang_analyzer_eval(i == 7); // expected-warning{{TRUE}}
}

// Nested lambdas.

void testNestedLambdas() {
  int i = 5;
  auto l = [i]() mutable {
    [&i]() {
      ++i;
    }();
    if (i != 6)
      clang_analyzer_warnIfReached();
  };
  l();
  clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
}

// Captured this.

class RandomClass {
  int i;

  void captureFields() {
    i = 5;
    [this]() {
      // clang_analyzer_eval does nothing in inlined functions.
      if (i != 5)
        clang_analyzer_warnIfReached();
      ++i;
    }();
    clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
  }
};


// Nested this capture.

class RandomClass2 {
  int i;

  void captureFields() {
    i = 5;
    [this]() {
      // clang_analyzer_eval does nothing in inlined functions.
      if (i != 5)
        clang_analyzer_warnIfReached();
      ++i;
      [this]() {
        // clang_analyzer_eval does nothing in inlined functions.
        if (i != 6)
          clang_analyzer_warnIfReached();
        ++i;
      }();
    }();
    clang_analyzer_eval(i == 7); // expected-warning{{TRUE}}
  }
};


// Captured function pointers.

void inc(int &x) {
  ++x;
}

void testFunctionPointerCapture() {
  void (*func)(int &) = inc;
  int i = 5;
  [&i, func] {
    func(i);
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}


// Test inline defensive checks
int getNum();

void inlineDefensiveChecks() {
  int i = getNum();
  [=]() {
    if (i == 0)
      ;
  }();
  int p = 5/i;
  (void)p;
}


template<typename T>
void callLambda(T t) {
  t();
}

struct DontCrash {
  int x;
  void f() {
    callLambda([&](){ ++x; });
    callLambdaFromStatic([&](){ ++x; });
  }
  
  template<typename T>
  static void callLambdaFromStatic(T t) {
    t();
  }
};


// Capture constants

void captureConstants() {
  const int i = 5;
  [=]() {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
}

void captureReferenceByCopy(int &p) {
  int v = 7;
  p = 8;

  // p is a reference captured by copy
  [&v,p]() mutable {
    v = p;
    p = 22;
  }();

  clang_analyzer_eval(v == 8); // expected-warning{{TRUE}}
  clang_analyzer_eval(p == 8); // expected-warning{{TRUE}}
}

void captureReferenceByReference(int &p) {
  int v = 7;
  p = 8;

  // p is a reference captured by reference
  [&v,&p]() {
    v = p;
    p = 22;
  }();

  clang_analyzer_eval(v == 8); // expected-warning{{TRUE}}
  clang_analyzer_eval(p == 22); // expected-warning{{TRUE}}
}

void callMutableLambdaMultipleTimes(int &p) {
  int v = 0;
  p = 8;

  auto l = [&v, p]() mutable {
    v = p;
    p++;
  };

  l();

  clang_analyzer_eval(v == 8); // expected-warning{{TRUE}}
  clang_analyzer_eval(p == 8); // expected-warning{{TRUE}}

  l();

  clang_analyzer_eval(v == 9); // expected-warning{{TRUE}}
  clang_analyzer_eval(p == 8); // expected-warning{{TRUE}}
}

// PR 24914
struct StructPR24914{
  int x;
};

void takesConstStructArgument(const StructPR24914&);
void captureStructReference(const StructPR24914& s) {
  [s]() {
    takesConstStructArgument(s);
  }();
}


// CHECK: [B2 (ENTRY)]
// CHECK:   Succs (1): B1
// CHECK: [B1]
// CHECK:   1: x
// CHECK:   2: [B1.1] (ImplicitCastExpr, NoOp, const struct X)
// CHECK:   3: [B1.2] (CXXConstructExpr, struct X)
// CHECK:   4: [x]     {
// CHECK:    }
// CHECK:   5: (void)[B1.4] (CStyleCastExpr, ToVoid, void)
// CHECK:   Preds (1): B2
// CHECK:   Succs (1): B0
// CHECK: [B0 (EXIT)]
// CHECK:   Preds (1): B1

