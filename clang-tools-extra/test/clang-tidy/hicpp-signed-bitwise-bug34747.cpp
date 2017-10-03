// RUN: %check_clang_tidy %s hicpp-signed-bitwise %t -- -- -std=c++11 | count 0

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
    o << 'x'; // warning occured here, fixed now
  }
};

void bar(OutputStream<signed char> &o) {
  foo<signed char> f(o);
}
