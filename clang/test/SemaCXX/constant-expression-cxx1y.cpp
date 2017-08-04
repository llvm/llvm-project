// RUN: %clang_cc1 -std=c++1y -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

struct S {
  // dummy ctor to make this a literal type
  constexpr S(int);

  S();

  int arr[10];

  constexpr int &get(int n) { return arr[n]; }
  constexpr const int &get(int n) const { return arr[n]; }
};

S s = S();
const S &sr = s;
static_assert(&s.get(4) - &sr.get(2) == 2, "");

// Compound-statements can be used in constexpr functions.
constexpr int e() {{{{}} return 5; }}
static_assert(e() == 5, "");

// Types can be defined in constexpr functions.
constexpr int f() {
  enum E { e1, e2, e3 };

  struct S {
    constexpr S(E e) : e(e) {}
    constexpr int get() { return e; }
    E e;
  };

  return S(e2).get();
}
static_assert(f() == 1, "");

// Variables can be declared in constexpr functions.
constexpr int g(int k) {
  const int n = 9;
  int k2 = k * k;
  int k3 = k2 * k;
  return 3 * k3 + 5 * k2 + n * k - 20;
}
static_assert(g(2) == 42, "");
constexpr int h(int n) {
  static const int m = n; // expected-error {{static variable not permitted in a constexpr function}}
  return m;
}
constexpr int i(int n) {
  thread_local const int m = n; // expected-error {{thread_local variable not permitted in a constexpr function}}
  return m;
}

// if-statements can be used in constexpr functions.
constexpr int j(int k) {
  if (k == 5)
    return 1;
  if (k == 1)
    return 5;
  else {
    if (int n = 2 * k - 4) {
      return n + 1;
      return 2;
    }
  }
} // expected-note 2{{control reached end of constexpr function}}
static_assert(j(0) == -3, "");
static_assert(j(1) == 5, "");
static_assert(j(2), ""); // expected-error {{constant expression}} expected-note {{in call to 'j(2)'}}
static_assert(j(3) == 3, "");
static_assert(j(4) == 5, "");
static_assert(j(5) == 1, "");

// There can be 0 return-statements.
constexpr void k() {
}

// If the return type is not 'void', no return statements => never a constant
// expression, so still diagnose that case.
[[noreturn]] constexpr int fn() { // expected-error {{no return statement in constexpr function}}
  fn();
}

// We evaluate the body of a constexpr constructor, to check for side-effects.
struct U {
  constexpr U(int n) {
    if (j(n)) {} // expected-note {{in call to 'j(2)'}}
  }
};
constexpr U u1{1};
constexpr U u2{2}; // expected-error {{constant expression}} expected-note {{in call to 'U(2)'}}

// We allow expression-statements.
constexpr int l(bool b) {
  if (b)
    throw "invalid value for b!"; // expected-note {{subexpression not valid}}
  return 5;
}
static_assert(l(false) == 5, "");
static_assert(l(true), ""); // expected-error {{constant expression}} expected-note {{in call to 'l(true)'}}

// Potential constant expression checking is still applied where possible.
constexpr int htonl(int x) { // expected-error {{never produces a constant expression}}
  typedef unsigned char uchar;
  uchar arr[4] = { uchar(x >> 24), uchar(x >> 16), uchar(x >> 8), uchar(x) };
  return *reinterpret_cast<int*>(arr); // expected-note {{reinterpret_cast is not allowed in a constant expression}}
}

constexpr int maybe_htonl(bool isBigEndian, int x) {
  if (isBigEndian)
    return x;

  typedef unsigned char uchar;
  uchar arr[4] = { uchar(x >> 24), uchar(x >> 16), uchar(x >> 8), uchar(x) };
  return *reinterpret_cast<int*>(arr); // expected-note {{reinterpret_cast is not allowed in a constant expression}}
}

constexpr int swapped = maybe_htonl(false, 123); // expected-error {{constant expression}} expected-note {{in call}}

namespace NS {
  constexpr int n = 0;
}
constexpr int namespace_alias() {
  namespace N = NS;
  return N::n;
}

namespace assign {
  constexpr int a = 0;
  const int b = 0;
  int c = 0; // expected-note {{here}}

  constexpr void set(const int &a, int b) {
    const_cast<int&>(a) = b; // expected-note 3{{constant expression cannot modify an object that is visible outside that expression}}
  }
  constexpr int wrap(int a, int b) {
    set(a, b);
    return a;
  }

  static_assert((set(a, 1), a) == 1, ""); // expected-error {{constant expression}} expected-note {{in call to 'set(a, 1)'}}
  static_assert((set(b, 1), b) == 1, ""); // expected-error {{constant expression}} expected-note {{in call to 'set(b, 1)'}}
  static_assert((set(c, 1), c) == 1, ""); // expected-error {{constant expression}} expected-note {{in call to 'set(c, 1)'}}

  static_assert(wrap(a, 1) == 1, "");
  static_assert(wrap(b, 1) == 1, "");
  static_assert(wrap(c, 1) == 1, ""); // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}
}

namespace string_assign {
  template<typename T>
  constexpr void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
  }
  template<typename Iterator>
  constexpr void reverse(Iterator begin, Iterator end) {
    while (begin != end && begin != --end)
      swap(*begin++, *end);
  }
  template<typename Iterator1, typename Iterator2>
  constexpr bool equal(Iterator1 a, Iterator1 ae, Iterator2 b, Iterator2 be) {
    while (a != ae && b != be)
      if (*a++ != *b++)
        return false;
    return a == ae && b == be;
  }
  constexpr bool test1(int n) {
    char stuff[100] = "foobarfoo";
    const char stuff2[100] = "oofraboof";
    reverse(stuff, stuff + n); // expected-note {{cannot refer to element 101 of array of 100 elements}}
    return equal(stuff, stuff + n, stuff2, stuff2 + n);
  }
  static_assert(!test1(1), "");
  static_assert(test1(3), "");
  static_assert(!test1(6), "");
  static_assert(test1(9), "");
  static_assert(!test1(100), "");
  static_assert(!test1(101), ""); // expected-error {{constant expression}} expected-note {{in call to 'test1(101)'}}

  constexpr void f() { // expected-error{{constexpr function never produces a constant expression}} expected-note@+2{{assignment to dereferenced one-past-the-end pointer is not allowed in a constant expression}}
    char foo[10] = { "z" }; // expected-note {{here}}
    foo[10] = 'x'; // expected-warning {{past the end}}
  }
}

namespace array_resize {
  constexpr int do_stuff(int k1, int k2) {
    int arr[1234] = { 1, 2, 3, 4 };
    arr[k1] = 5; // expected-note {{past-the-end}} expected-note {{cannot refer to element 1235}} expected-note {{cannot refer to element -1}}
    return arr[k2];
  }
  static_assert(do_stuff(1, 2) == 3, "");
  static_assert(do_stuff(0, 0) == 5, "");
  static_assert(do_stuff(1233, 1233) == 5, "");
  static_assert(do_stuff(1233, 0) == 1, "");
  static_assert(do_stuff(1234, 0) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
  static_assert(do_stuff(1235, 0) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
  static_assert(do_stuff(-1, 0) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace potential_const_expr {
  constexpr void set(int &n) { n = 1; }
  constexpr int div_zero_1() { int z = 0; set(z); return 100 / z; } // no error
  constexpr int div_zero_2() { // expected-error {{never produces a constant expression}}
    int z = 0;
    return 100 / (set(z), 0); // expected-note {{division by zero}}
  }
  int n; // expected-note {{declared here}}
  constexpr int ref() { // expected-error {{never produces a constant expression}}
    int &r = n;
    return r; // expected-note {{read of non-const variable 'n'}}
  }
}

namespace subobject {
  union A { constexpr A() : y(5) {} int x, y; };
  struct B { A a; };
  struct C : B {};
  union D { constexpr D() : c() {} constexpr D(int n) : n(n) {} C c; int n; };
  constexpr void f(D &d) {
    d.c.a.y = 3;
    // expected-note@-1 {{cannot modify an object that is visible outside}}
    // expected-note@-2 {{assignment to member 'c' of union with active member 'n'}}
  }
  constexpr bool check(D &d) { return d.c.a.y == 3; }

  constexpr bool g() { D d; f(d); return d.c.a.y == 3; }
  static_assert(g(), "");

  D d;
  constexpr bool h() { f(d); return check(d); } // expected-note {{in call}}
  static_assert(h(), ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr bool i() { D d(0); f(d); return check(d); } // expected-note {{in call}}
  static_assert(i(), ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr bool j() { D d; d.c.a.x = 3; return check(d); } // expected-note {{assignment to member 'x' of union with active member 'y'}}
  static_assert(j(), ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace lifetime {
  constexpr int &&id(int &&n) { return static_cast<int&&>(n); }
  constexpr int &&dead() { return id(0); } // expected-note {{temporary created here}}
  constexpr int bad() { int &&n = dead(); n = 1; return n; } // expected-note {{assignment to temporary whose lifetime has ended}}
  static_assert(bad(), ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace const_modify {
  constexpr int modify(int &n) { return n = 1; } // expected-note 2 {{modification of object of const-qualified type 'const int'}}
  constexpr int test1() { int k = 0; return modify(k); }
  constexpr int test2() { const int k = 0; return modify(const_cast<int&>(k)); } // expected-note 2 {{in call}}
  static_assert(test1() == 1, "");
  static_assert(test2() == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}
  constexpr int i = test2(); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace null {
  constexpr int test(int *p) {
    return *p = 123; // expected-note {{assignment to dereferenced null pointer}}
  }
  static_assert(test(0), ""); // expected-error {{constant expression}} expected-note {{in call}}
}

namespace incdec {
  template<typename T> constexpr T &ref(T &&r) { return r; }
  template<typename T> constexpr T postinc(T &&r) { return (r++, r); }
  template<typename T> constexpr T postdec(T &&r) { return (r--, r); }

  static_assert(++ref(0) == 1, "");
  static_assert(ref(0)++ == 0, "");
  static_assert(postinc(0) == 1, "");
  static_assert(--ref(0) == -1, "");
  static_assert(ref(0)-- == 0, "");
  static_assert(postdec(0) == -1, "");

  constexpr int overflow_int_inc_1 = ref(0x7fffffff)++; // expected-error {{constant}} expected-note {{2147483648}}
  constexpr int overflow_int_inc_1_ok = ref(0x7ffffffe)++;
  constexpr int overflow_int_inc_2 = ++ref(0x7fffffff); // expected-error {{constant}} expected-note {{2147483648}}
  constexpr int overflow_int_inc_2_ok = ++ref(0x7ffffffe);

  // inc/dec on short can't overflow because we promote to int first
  static_assert(++ref<short>(0x7fff) == (int)0xffff8000u, "");
  static_assert(--ref<short>(0x8000) == 0x7fff, "");

  // inc on bool sets to true
  static_assert(++ref(false), ""); // expected-warning {{deprecated}}
  static_assert(++ref(true), ""); // expected-warning {{deprecated}}

  int arr[10];
  static_assert(++ref(&arr[0]) == &arr[1], "");
  static_assert(++ref(&arr[9]) == &arr[10], "");
  static_assert(++ref(&arr[10]) == &arr[11], ""); // expected-error {{constant}} expected-note {{cannot refer to element 11}}
  static_assert(ref(&arr[0])++ == &arr[0], "");
  static_assert(ref(&arr[10])++ == &arr[10], ""); // expected-error {{constant}} expected-note {{cannot refer to element 11}}
  static_assert(postinc(&arr[0]) == &arr[1], "");
  static_assert(--ref(&arr[10]) == &arr[9], "");
  static_assert(--ref(&arr[1]) == &arr[0], "");
  static_assert(--ref(&arr[0]) != &arr[0], ""); // expected-error {{constant}} expected-note {{cannot refer to element -1}}
  static_assert(ref(&arr[1])-- == &arr[1], "");
  static_assert(ref(&arr[0])-- == &arr[0], ""); // expected-error {{constant}} expected-note {{cannot refer to element -1}}
  static_assert(postdec(&arr[1]) == &arr[0], "");

  int x;
  static_assert(++ref(&x) == &x + 1, "");

  static_assert(++ref(0.0) == 1.0, "");
  static_assert(ref(0.0)++ == 0.0, "");
  static_assert(postinc(0.0) == 1.0, "");
  static_assert(--ref(0.0) == -1.0, "");
  static_assert(ref(0.0)-- == 0.0, "");
  static_assert(postdec(0.0) == -1.0, "");

  static_assert(++ref(1e100) == 1e100, "");
  static_assert(--ref(1e100) == 1e100, "");

  union U {
    int a, b;
  };
  constexpr int f(U u) {
    return ++u.b; // expected-note {{increment of member 'b' of union with active member 'a'}}
  }
  constexpr int wrong_member = f({0}); // expected-error {{constant}} expected-note {{in call to 'f({.a = 0})'}}
  constexpr int vol = --ref<volatile int>(0); // expected-error {{constant}} expected-note {{decrement of volatile-qualified}}

  constexpr int incr(int k) {
    int x = k;
    if (x++ == 100)
      return x;
    return incr(x);
  }
  static_assert(incr(0) == 101, "");
}

namespace compound_assign {
  constexpr bool test_int() {
    int a = 3;
    a += 6;
    if (a != 9) return false;
    a -= 2;
    if (a != 7) return false;
    a *= 3;
    if (a != 21) return false;
    if (&(a /= 10) != &a) return false;
    if (a != 2) return false;
    a <<= 3;
    if (a != 16) return false;
    a %= 6;
    if (a != 4) return false;
    a >>= 1;
    if (a != 2) return false;
    a ^= 10;
    if (a != 8) return false;
    a |= 5;
    if (a != 13) return false;
    a &= 14;
    if (a != 12) return false;
    return true;
  }
  static_assert(test_int(), "");

  constexpr bool test_float() {
    float f = 123.;
    f *= 2;
    if (f != 246.) return false;
    if ((f -= 0.5) != 245.5) return false;
    if (f != 245.5) return false;
    f /= 0.5;
    if (f != 491.) return false;
    f += -40;
    if (f != 451.) return false;
    return true;
  }
  static_assert(test_float(), "");

  constexpr bool test_ptr() {
    int arr[123] = {};
    int *p = arr;
    if ((p += 4) != &arr[4]) return false;
    if (p != &arr[4]) return false;
    p += -1;
    if (p != &arr[3]) return false;
    if ((p -= -10) != &arr[13]) return false;
    if (p != &arr[13]) return false;
    p -= 11;
    if (p != &arr[2]) return false;
    return true;
  }
  static_assert(test_ptr(), "");

  template<typename T>
  constexpr bool test_overflow() {
    T a = 1;
    while (a != a / 2)
      a *= 2; // expected-note {{value 2147483648 is outside the range}} expected-note {{ 9223372036854775808 }} expected-note {{floating point arithmetic produces an infinity}}
    return true;
  }

  static_assert(test_overflow<int>(), ""); // expected-error {{constant}} expected-note {{call}}
  static_assert(test_overflow<unsigned>(), ""); // ok, unsigned overflow is defined
  static_assert(test_overflow<short>(), ""); // ok, short is promoted to int before multiplication
  static_assert(test_overflow<unsigned short>(), ""); // ok
  static_assert(test_overflow<unsigned long long>(), ""); // ok
  static_assert(test_overflow<long long>(), ""); // expected-error {{constant}} expected-note {{call}}
  static_assert(test_overflow<float>(), ""); // expected-error {{constant}} expected-note {{call}}

  constexpr short test_promotion(short k) {
    short s = k;
    s *= s;
    return s;
  }
  static_assert(test_promotion(100) == 10000, "");
  static_assert(test_promotion(200) == -25536, "");
  static_assert(test_promotion(256) == 0, "");

  constexpr const char *test_bounds(const char *p, int o) {
    return p += o; // expected-note {{element 5 of}} expected-note {{element -1 of}} expected-note {{element 1000 of}}
  }
  static_assert(test_bounds("foo", 0)[0] == 'f', "");
  static_assert(test_bounds("foo", 3)[0] == 0, "");
  static_assert(test_bounds("foo", 4)[-3] == 'o', "");
  static_assert(test_bounds("foo" + 4, -4)[0] == 'f', "");
  static_assert(test_bounds("foo", 5) != 0, ""); // expected-error {{constant}} expected-note {{call}}
  static_assert(test_bounds("foo", -1) != 0, ""); // expected-error {{constant}} expected-note {{call}}
  static_assert(test_bounds("foo", 1000) != 0, ""); // expected-error {{constant}} expected-note {{call}}
}

namespace loops {
  constexpr int fib_loop(int a) {
    int f_k = 0, f_k_plus_one = 1;
    for (int k = 1; k != a; ++k) {
      int f_k_plus_two = f_k + f_k_plus_one;
      f_k = f_k_plus_one;
      f_k_plus_one = f_k_plus_two;
    }
    return f_k_plus_one;
  }
  static_assert(fib_loop(46) == 1836311903, "");

  constexpr bool breaks_work() {
    int a = 0;
    for (int n = 0; n != 100; ++n) {
      ++a;
      if (a == 5) continue;
      if ((a % 5) == 0) break;
    }

    int b = 0;
    while (b != 17) {
      ++b;
      if (b == 6) continue;
      if ((b % 6) == 0) break;
    }

    int c = 0;
    do {
      ++c;
      if (c == 7) continue;
      if ((c % 7) == 0) break;
    } while (c != 21);

    return a == 10 && b == 12 && c == 14;
  }
  static_assert(breaks_work(), "");

  void not_constexpr();
  constexpr bool no_cont_after_break() {
    for (;;) {
      break;
      not_constexpr();
    }
    while (true) {
      break;
      not_constexpr();
    }
    do {
      break;
      not_constexpr();
    } while (true);
    return true;
  }
  static_assert(no_cont_after_break(), "");

  constexpr bool cond() {
    for (int a = 1; bool b = a != 3; ++a) {
      if (!b)
        return false;
    }
    while (bool b = true) {
      b = false;
      break;
    }
    return true;
  }
  static_assert(cond(), "");

  constexpr int range_for() {
    int arr[] = { 1, 2, 3, 4, 5 };
    int sum = 0;
    for (int x : arr)
      sum += x;
    return sum;
  }
  static_assert(range_for() == 15, "");

  template<int...N> struct ints {};
  template<typename A, typename B> struct join_ints;
  template<int...As, int...Bs> struct join_ints<ints<As...>, ints<Bs...>> {
    using type = ints<As..., sizeof...(As) + Bs...>;
  };
  template<unsigned N> struct make_ints {
    using type = typename join_ints<typename make_ints<N/2>::type, typename make_ints<(N+1)/2>::type>::type;
  };
  template<> struct make_ints<0> { using type = ints<>; };
  template<> struct make_ints<1> { using type = ints<0>; };

  struct ignore { template<typename ...Ts> constexpr ignore(Ts &&...) {} };

  template<typename T, unsigned N> struct array {
    constexpr array() : arr{} {}
    template<typename ...X>
    constexpr array(X ...x) : arr{} {
      init(typename make_ints<sizeof...(X)>::type{}, x...);
    }
    template<int ...I, typename ...X> constexpr void init(ints<I...>, X ...x) {
      ignore{arr[I] = x ...};
    }
    T arr[N];
    struct iterator {
      T *p;
      constexpr explicit iterator(T *p) : p(p) {}
      constexpr bool operator!=(iterator o) { return p != o.p; }
      constexpr iterator &operator++() { ++p; return *this; }
      constexpr T &operator*() { return *p; }
    };
    constexpr iterator begin() { return iterator(arr); }
    constexpr iterator end() { return iterator(arr + N); }
  };

  constexpr int range_for_2() {
    array<int, 5> arr { 1, 2, 3, 4, 5 };
    int sum = 0;
    for (int k : arr) {
      sum += k;
      if (sum > 8) break;
    }
    return sum;
  }
  static_assert(range_for_2() == 10, "");
}

namespace assignment_op {
  struct A {
    constexpr A() : n(5) {}
    int n;
    struct B {
      int k = 1;
      union U {
        constexpr U() : y(4) {}
        int x;
        int y;
      } u;
    } b;
  };
  constexpr bool testA() {
    A a, b;
    a.n = 7;
    a.b.u.y = 5;
    b = a;
    return b.n == 7 && b.b.u.y == 5 && b.b.k == 1;
  }
  static_assert(testA(), "");

  struct B {
    bool assigned = false;
    constexpr B &operator=(const B&) {
      assigned = true;
      return *this;
    }
  };
  struct C : B {
    B b;
    int n = 5;
  };
  constexpr bool testC() {
    C c, d;
    c.n = 7;
    d = c;
    c.n = 3;
    return d.n == 7 && d.assigned && d.b.assigned;
  }
  static_assert(testC(), "");
}

namespace switch_stmt {
  constexpr int f(char k) {
    bool b = false;
    int z = 6;
    switch (k) {
      return -1;
    case 0:
      if (false) {
      case 1:
        z = 1;
        for (; b;) {
          return 5;
          while (0)
            case 2: return 2;
          case 7: z = 7;
          do case 6: {
            return z;
            if (false)
              case 3: return 3;
            case 4: z = 4;
          } while (1);
          case 5: b = true;
          case 9: z = 9;
        }
        return z;
      } else if (false) case 8: z = 8;
      else if (false) {
      case 10:
        z = -10;
        break;
      }
      else z = 0;
      return z;
    default:
      return -1;
    }
    return -z;
  }
  static_assert(f(0) == 0, "");
  static_assert(f(1) == 1, "");
  static_assert(f(2) == 2, "");
  static_assert(f(3) == 3, "");
  static_assert(f(4) == 4, "");
  static_assert(f(5) == 5, "");
  static_assert(f(6) == 6, "");
  static_assert(f(7) == 7, "");
  static_assert(f(8) == 8, "");
  static_assert(f(9) == 9, "");
  static_assert(f(10) == 10, "");

  // Check that we can continue an outer loop from within a switch.
  constexpr bool contin() {
    for (int n = 0; n != 10; ++n) {
      switch (n) {
      case 0:
        ++n;
        continue;
      case 1:
        return false;
      case 2:
        return true;
      }
    }
    return false;
  }
  static_assert(contin(), "");

  constexpr bool switch_into_for() {
    int n = 0;
    switch (n) {
      for (; n == 1; ++n) {
        return n == 1;
      case 0: ;
      }
    }
    return false;
  }
  static_assert(switch_into_for(), "");

  constexpr void duff_copy(char *a, const char *b, int n) {
    switch ((n - 1) % 8 + 1) {
      for ( ; n; n = (n - 1) & ~7) {
      case 8: a[n-8] = b[n-8];
      case 7: a[n-7] = b[n-7];
      case 6: a[n-6] = b[n-6];
      case 5: a[n-5] = b[n-5];
      case 4: a[n-4] = b[n-4];
      case 3: a[n-3] = b[n-3];
      case 2: a[n-2] = b[n-2];
      case 1: a[n-1] = b[n-1];
      }
      case 0: ;
    }
  }

  constexpr bool test_copy(const char *str, int n) {
    char buffer[16] = {};
    duff_copy(buffer, str, n);
    for (int i = 0; i != sizeof(buffer); ++i)
      if (buffer[i] != (i < n ? str[i] : 0))
        return false;
    return true;
  }
  static_assert(test_copy("foo", 0), "");
  static_assert(test_copy("foo", 1), "");
  static_assert(test_copy("foo", 2), "");
  static_assert(test_copy("hello world", 0), "");
  static_assert(test_copy("hello world", 7), "");
  static_assert(test_copy("hello world", 8), "");
  static_assert(test_copy("hello world", 9), "");
  static_assert(test_copy("hello world", 10), "");
  static_assert(test_copy("hello world", 10), "");
}

namespace deduced_return_type {
  constexpr auto f() { return 0; }
  template<typename T> constexpr auto g(T t) { return t; }
  static_assert(f() == 0, "");
  static_assert(g(true), "");
}

namespace modify_temporary_during_construction {
  struct A { int &&temporary; int x; int y; };
  constexpr int f(int &r) { r *= 9; return r - 12; }
  constexpr A a = { 6, f(a.temporary), a.temporary }; // expected-note {{temporary created here}}
  static_assert(a.x == 42, "");
  static_assert(a.y == 54, "");
  constexpr int k = a.temporary++; // expected-error {{constant expression}} expected-note {{outside the expression that created the temporary}}
}

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    constexpr initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    constexpr initializer_list() : __begin_(nullptr), __size_(0) {}

    constexpr size_t    size()  const {return __size_;}
    constexpr const _E* begin() const {return __begin_;}
    constexpr const _E* end()   const {return __begin_ + __size_;}
  };
}

namespace InitializerList {
  constexpr int sum(std::initializer_list<int> ints) {
    int total = 0;
    for (int n : ints) total += n;
    return total;
  }
  static_assert(sum({1, 2, 3, 4, 5}) == 15, "");
}

namespace StmtExpr {
  constexpr int f(int k) {
    switch (k) {
    case 0:
      return 0;

      ({
        case 1: // expected-note {{not supported}}
          return 1;
      });
    }
  }
  static_assert(f(1) == 1, ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr int g() { // expected-error {{never produces a constant}}
    return ({ int n; n; }); // expected-note {{object of type 'int' is not initialized}}
  }

  // FIXME: We should handle the void statement expression case.
  constexpr int h() { // expected-error {{never produces a constant}}
    ({ if (true) {} }); // expected-note {{not supported}}
    return 0;
  }
}

namespace VirtualFromBase {
  struct S1 {
    virtual int f() const;
  };
  struct S2 {
    virtual int f();
  };
  template <typename T> struct X : T {
    constexpr X() {}
    double d = 0.0;
    constexpr int f() { return sizeof(T); }
  };

  // Non-virtual f(), OK.
  constexpr X<X<S1>> xxs1;
  constexpr X<S1> *p = const_cast<X<X<S1>>*>(&xxs1);
  static_assert(p->f() == sizeof(S1), "");

  // Virtual f(), not OK.
  constexpr X<X<S2>> xxs2;
  constexpr X<S2> *q = const_cast<X<X<S2>>*>(&xxs2);
  static_assert(q->f() == sizeof(X<S2>), ""); // expected-error {{constant expression}} expected-note {{virtual function call}}
}

namespace Lifetime {
  constexpr int &get(int &&r) { return r; }
  constexpr int f() {
    int &r = get(123);
    return r; // expected-note {{read of object outside its lifetime}}
  }
  static_assert(f() == 123, ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr int g() {
    int *p = 0;
    {
      int n = 0;
      p = &n;
      n = 42;
    }
    *p = 123; // expected-note {{assignment to object outside its lifetime}}
    return *p;
  }
  static_assert(g() == 42, ""); // expected-error {{constant expression}} expected-note {{in call}}

  constexpr int h(int n) {
    int *p[4] = {};
    int &&r = 1;
    p[0] = &r;
    while (int a = 1) {
      p[1] = &a;
      for (int b = 1; int c = 1; ) {
        p[2] = &b, p[3] = &c;
        break;
      }
      break;
    }
    *p[n] = 0; // expected-note 3{{assignment to object outside its lifetime}}
    return *p[n];
  }
  static_assert(h(0) == 0, ""); // ok, lifetime-extended
  static_assert(h(1) == 0, ""); // expected-error {{constant expression}} expected-note {{in call}}
  static_assert(h(2) == 0, ""); // expected-error {{constant expression}} expected-note {{in call}}
  static_assert(h(3) == 0, ""); // expected-error {{constant expression}} expected-note {{in call}}

  // FIXME: This function should be treated as non-constant.
  constexpr void lifetime_versus_loops() {
    int *p = 0;
    for (int i = 0; i != 2; ++i) {
      int *q = p;
      int n = 0;
      p = &n;
      if (i)
        // This modifies the 'n' from the previous iteration of the loop outside
        // its lifetime.
        ++*q;
    }
  }
  static_assert((lifetime_versus_loops(), true), "");
}

namespace Bitfields {
  struct A {
    bool b : 1;
    int n : 4;
    unsigned u : 5;
  };
  constexpr bool test() {
    A a {};
    a.b += 2;
    --a.n;
    --a.u;
    a.n = -a.n * 3;
    return a.b == false && a.n == 3 && a.u == 31;
  }
  static_assert(test(), "");
}

namespace PR17615 {
  struct A {
    int &&r;
    constexpr A(int &&r) : r(static_cast<int &&>(r)) {}
    constexpr A() : A(0) {
      (void)+r; // expected-note {{outside its lifetime}}
    }
  };
  constexpr int k = A().r; // expected-error {{constant expression}} expected-note {{in call to}}
}

namespace PR17331 {
  template<typename T, unsigned int N>
  constexpr T sum(const T (&arr)[N]) {
    T result = 0;
    for (T i : arr)
      result += i;
    return result;
  }

  constexpr int ARR[] = { 1, 2, 3, 4, 5 };
  static_assert(sum(ARR) == 15, "");
}

namespace EmptyClass {
  struct E1 {} e1;
  union E2 {} e2; // expected-note 4{{here}}
  struct E3 : E1 {} e3;

  template<typename E>
  constexpr int f(E &a, int kind) {
    switch (kind) {
    case 0: { E e(a); return 0; } // expected-note {{read}} expected-note {{in call}}
    case 1: { E e(static_cast<E&&>(a)); return 0; } // expected-note {{read}} expected-note {{in call}}
    case 2: { E e; e = a; return 0; } // expected-note {{read}} expected-note {{in call}}
    case 3: { E e; e = static_cast<E&&>(a); return 0; } // expected-note {{read}} expected-note {{in call}}
    }
  }
  constexpr int test1 = f(e1, 0);
  constexpr int test2 = f(e2, 0); // expected-error {{constant expression}} expected-note {{in call}}
  constexpr int test3 = f(e3, 0);
  constexpr int test4 = f(e1, 1);
  constexpr int test5 = f(e2, 1); // expected-error {{constant expression}} expected-note {{in call}}
  constexpr int test6 = f(e3, 1);
  constexpr int test7 = f(e1, 2);
  constexpr int test8 = f(e2, 2); // expected-error {{constant expression}} expected-note {{in call}}
  constexpr int test9 = f(e3, 2);
  constexpr int testa = f(e1, 3);
  constexpr int testb = f(e2, 3); // expected-error {{constant expression}} expected-note {{in call}}
  constexpr int testc = f(e3, 3);
}

namespace SpeculativeEvalWrites {
  // Ensure that we don't try to speculatively evaluate writes.
  constexpr int f() {
    int i = 0;
    int a = 0;
    // __builtin_object_size speculatively evaluates its first argument.
    __builtin_object_size((i = 1, &a), 0);
    return i;
  }

  static_assert(!f(), "");
}

namespace PR27989 {
  constexpr int f(int n) {
    int a = (n = 1, 0);
    return n;
  }
  static_assert(f(0) == 1, "");
}

namespace const_char {
template <int N>
constexpr int sum(const char (&Arr)[N]) {
  int S = 0;
  for (unsigned I = 0; I != N; ++I)
    S += Arr[I]; // expected-note 2{{read of non-constexpr variable 'Cs' is not allowed}}
  return S;
}

// As an extension, we support evaluating some things that are `const` as though
// they were `constexpr` when folding, but it should not be allowed in normal
// constexpr evaluation.
const char Cs[] = {'a', 'b'};
void foo() __attribute__((enable_if(sum(Cs) == 'a' + 'b', "")));
void run() { foo(); }

static_assert(sum(Cs) == 'a' + 'b', ""); // expected-error{{not an integral constant expression}} expected-note{{in call to 'sum(Cs)'}}
constexpr int S = sum(Cs); // expected-error{{must be initialized by a constant expression}} expected-note{{in call}}
}

constexpr void PR28739(int n) { // expected-error {{never produces a constant}}
  int *p = &n;
  p += (__int128)(unsigned long)-1; // expected-note {{cannot refer to element 18446744073709551615 of non-array object in a constant expression}}
}
