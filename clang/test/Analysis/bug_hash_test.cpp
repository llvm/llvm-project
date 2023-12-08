// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,debug.ExprInspection %s -verify

constexpr int clang_analyzer_hashDump(int) { return 5; }

void function(int) {
  clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void function(int)$27$clang_analyzer_hashDump(5);$Category}}
}

namespace {
void variadicParam(int, ...) {
  clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void (anonymous namespace)::variadicParam(int, ...)$27$clang_analyzer_hashDump(5);$Category}}
}
} // namespace

constexpr int f() {
  return clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$int f()$34$returnclang_analyzer_hashDump(5);$Category}}
}

namespace AA {
class X {
  X() {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$AA::X::X()$29$clang_analyzer_hashDump(5);$Category}}
  }

  static void static_method() {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void AA::X::static_method()$29$clang_analyzer_hashDump(5);$Category}}
    variadicParam(5);
  }

  void method() && {
    struct Y {
      inline void method() const & {
        clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void AA::X::method()::Y::method() const &$33$clang_analyzer_hashDump(5);$Category}}
      }
    };

    Y y;
    y.method();

    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void AA::X::method() &&$29$clang_analyzer_hashDump(5);$Category}}
  }

  void OutOfLine();

  X &operator=(int) {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$X & AA::X::operator=(int)$29$clang_analyzer_hashDump(5);$Category}}
    return *this;
  }

  operator int() {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$AA::X::operator int()$29$clang_analyzer_hashDump(5);$Category}}
    return 0;
  }

  explicit operator float() {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$AA::X::operator float()$29$clang_analyzer_hashDump(5);$Category}}
    return 0;
  }
};
} // namespace AA

void AA::X::OutOfLine() {
  clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void AA::X::OutOfLine()$27$clang_analyzer_hashDump(5);$Category}}
}

void testLambda() {
  []() {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void testLambda()::(anonymous class)::operator()() const$29$clang_analyzer_hashDump(5);$Category}}
  }();
}

template <typename T>
void f(T) {
  clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void f(T)$27$clang_analyzer_hashDump(5);$Category}}
}

template <typename T>
struct TX {
  void f(T) {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void TX::f(T)$29$clang_analyzer_hashDump(5);$Category}}
  }
};

template <>
void f<long>(long) {
  clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void f(long)$27$clang_analyzer_hashDump(5);$Category}}
}

template <>
struct TX<long> {
  void f(long) {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void TX<long>::f(long)$29$clang_analyzer_hashDump(5);$Category}}
  }
};

template <typename T>
struct TTX {
  template<typename S>
  void f(T, S) {
    clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$void TTX::f(T, S)$29$clang_analyzer_hashDump(5);$Category}}
  }
};

void g() {
  // TX<int> and TX<double> is instantiated from the same code with the same
  // source locations. The same error happining in both of the instantiations
  // should share the common hash. This means we should not include the
  // template argument for these types in the function signature.
  // Note that, we still want the hash to be different for explicit
  // specializations.
  TX<int> x;
  TX<double> y;
  TX<long> xl;
  x.f(1);
  xl.f(1);
  f(5);
  f(3.0);
  y.f(2);
  TTX<int> z;
  z.f<int>(5, 5);
  f(5l);
}
