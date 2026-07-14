// RUN: %clang_cc1 %std_cxx11- -fexceptions -fcxx-exceptions -fsyntax-only --embed-dir=%S/../../../../Preprocessor/Inputs -Wno-c23-extensions -verify %s

namespace std {
using size_t = decltype(sizeof(int));

template <class E> class initializer_list {
  const E *begin_;
  size_t size_;

public:
  constexpr initializer_list() : begin_(nullptr), size_(0) {}
  constexpr initializer_list(const E *begin, size_t size)
      : begin_(begin), size_(size) {}
  constexpr const E *begin() const { return begin_; }
  constexpr const E *end() const { return begin_ + size_; }
  constexpr size_t size() const { return size_; }
};

template <class T> struct complex {
  constexpr complex(double) {}
};

template <class T> struct vector {
  vector(initializer_list<T>);
};
} // namespace std

namespace example12 {
void f(std::initializer_list<double> il);
void g(float x) {
  f({1, x, 3});
}
void h() {
  f({1, 2, 3});
}

struct A {
  mutable int i;
};
void q(std::initializer_list<A>);
void r() {
  q({A{1}, A{2}, A{3}});
}
} // namespace example12

namespace example13 {
typedef std::complex<double> cmplx;
std::vector<cmplx> v1 = {1, 2, 3};
void f() {
  std::vector<cmplx> v2{1, 2, 3};
  std::initializer_list<int> i3 = {1, 2, 3};
}

struct A {
  std::initializer_list<int> i4; // expected-note {{'std::initializer_list' member declared here}}
  A() : i4{1, 2, 3} {}
  // expected-error@-1 {{backing array for 'std::initializer_list' member 'i4' is a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
};
} // namespace example13

namespace embed_example {
void bytes(std::initializer_list<unsigned char>);
void f() {
  bytes({
#embed <jk.txt>
  });
}

constexpr std::initializer_list<unsigned char> jk = {
#embed <jk.txt>
};
static_assert(jk.size() == 2, "");
static_assert(jk.begin()[0] == 'j', "");
static_assert(jk.begin()[1] == 'k', "");
} // namespace embed_example

namespace shared_backing_arrays {
void f2(std::initializer_list<int> ia, std::initializer_list<int> ib) {
  (void)(ia.begin() == ib.begin());
}
void test() {
  f2({1, 2, 3}, {1, 2, 3});
}

void f3() {
  std::initializer_list<int> i1 = {1, 2, 3, 4, 5};
  std::initializer_list<int> i2 = {2, 3, 4};
  (void)(i1.begin() == i2.begin() + 1);
}
} // namespace shared_backing_arrays

namespace lifetime_is_unchanged {
const int *f4(std::initializer_list<int> i4) {
  return i4.begin();
}
void test() {
  const int *p = f4({1, 2, 3});
  (void)*p;
}
} // namespace lifetime_is_unchanged

namespace destructor_side_effects {
extern "C" int printf(const char *, ...);

struct C6 {
  constexpr C6(int) {}
  ~C6() { printf(" X"); }
};

void f6(std::initializer_list<C6>) {}
void test() {
  f6({1, 2, 3});
  f6({1, 2, 3});
}
} // namespace destructor_side_effects

namespace mutable_members {
struct S {
  constexpr S(int i) : i(i) {}
  mutable int i;
};

void f(std::initializer_list<S> il) {
  if (il.begin()->i != 1)
    throw;
  il.begin()->i = 4;
}
void test() {
  for (int i = 0; i < 2; ++i)
    f({1, 2, 3});
}
} // namespace mutable_members

namespace annex_c {
bool ne(std::initializer_list<int> a, std::initializer_list<int> b) {
  return a.begin() != b.begin() + 1;
}
bool b = ne({2, 3}, {1, 2, 3});
} // namespace annex_c
