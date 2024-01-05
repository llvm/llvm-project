// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {
  int a, b[3], c;
  bool operator==(const A&) const = default;
};

constexpr auto a0 = A{0, 0, 3, 4, 5};

// expected-note@+1 {{evaluates to '{0, {0, 3, 4}, 5} == {1, {2, 3, 4}, 5}'}}
static_assert(a0 == A{1, {2, 3, 4}, 5}); // expected-error {{failed}}

struct _arr {
  const int b[3];
  constexpr bool operator==(const int rhs[3]) const {
    for (unsigned i = 0; i < sizeof(b) / sizeof(int); i++)
      if (b[i] != rhs[i])
        return false;
    return true;
  }
};

// output: '{{2, 3, 4}} == {0, 3, 4}'  (the `{{` breaks VerifyDiagnosticConsumer::ParseDirective)
// expected-note@+1 {{evaluates to}}
static_assert(_arr{2, 3, 4} == a0.b); // expected-error {{failed}}

struct B {
  int a, c; // named the same just to keep things fresh
  bool operator==(const B&) const = default;
};

// expected-note@+1 {{evaluates to '{7, 6} == {8, 6}'}}
static_assert(B{7, 6} == B{8, 6}); // expected-error {{failed}}

typedef int v4si __attribute__((__vector_size__(16)));

struct C: A, B {
  enum { E1, E2 } e;
  bool operator==(const C&) const = default;
};

constexpr auto cc = C{A{1, {2, 3, 4}, 5}, B{7, 6}, C::E1};

// actually '{{1, {2, 3, 4}, 5}, {7, 6}, 0} == {{0, {0, 3, 4}, 5}, {5, 0}, 1}'  (the `{{` breaks VerifyDiagnosticConsumer::ParseDirective)
// expected-note@+1 {{evaluates to}}
static_assert(cc == C{a0, {5}, C::E2}); // expected-error {{failed}}

// this little guy? oh, I wouldn't worry about this little guy
namespace std {
template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, from);
}
} // namespace std

typedef int v4si __attribute__((__vector_size__(16)));

struct V {
  v4si v;

  // doesn't work
  // vectors are not contextually convertable to `bool`, and
  // `==` on vectors produces a vector of element-wise results
  // bool operator==(const V&) const = default;

  constexpr bool operator==(const V& rhs) const {
    // doesn't work
    // __builtin_reduce_and is not valid in a constant expression
    // return __builtin_reduce_and(b == rhs.b) && __builtin_reduce_and(v == rhs.v);

    // also doesn't work
    // surprisingly, b[0] is also not valid in a constant expression (nor v[0])
    // return b[0] == rhs.b[0] && ...

    struct cmp {
      unsigned char v [sizeof(v4si)];
      bool operator==(const cmp&) const = default;
    };
    return std::bit_cast<cmp>(v) == std::bit_cast<cmp>(rhs.v);
  };

};
static_assert(V{1, 2, 3, 4} == V{1, 2, 3, 4});

// '{{1, 2, 3, 4}} == {{1, 2, 0, 4}}'
// expected-note@+1 {{evaluates to}}
static_assert(V{1, 2, 3, 4} == V{1, 2, 0, 4}); // expected-error {{failed}}

constexpr auto v = (v4si){1, 2, 3, 4};
constexpr auto vv = V{{1, 2, 3, 4}};


// there appears to be no constexpr-compatible way to write an == for
// two `bool4`s at this time, since std::bit_cast doesn't support it
// typedef bool bool4 __attribute__((ext_vector_type(4)));

// so we use a bool8
typedef bool bool8 __attribute__((ext_vector_type(8)));

struct BV {
  bool8 b;
  constexpr bool operator==(const BV& rhs) const {
    return std::bit_cast<unsigned char>(b) == std::bit_cast<unsigned char>(rhs.b);
  }
};

// '{{false, true, false, false, false, false, false, false}} == {{true, false, false, false, false, false, false, false}}'
// expected-note@+1 {{evaluates to}}
static_assert(BV{{0, 1}} == BV{{1, 0}}); // expected-error {{failed}}
