// RUN: %check_clang_tidy %s hicpp-signed-bitwise %t --

// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.

template <typename C>
struct OutputStream {
  OutputStream &operator<<(C);
};

template <typename C>
struct foo {
  typedef OutputStream<C> stream_type;
  foo(stream_type &o) {
    o << 'x'; // warning occurred here, fixed now
  }
};

void bar(OutputStream<signed char> &o) {
  foo<signed char> f(o);
}

void silence_lit() {
  int SValue = 42;
  int SResult;

  SResult = SValue & 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
}
