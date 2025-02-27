// RUN: %clang_cc1 -fsyntax-only -Wall -Wc++20-compat -Wuninitialized -Wno-unused-value -Wno-unused-lambda-capture -Wno-uninitialized-const-reference -std=c++1z -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wall -Wc++20-compat -Wuninitialized -Wno-unused-value -Wno-unused-lambda-capture -Wno-uninitialized-const-reference -std=c++1z -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -Wall -Wc++20-compat -Wuninitialized -Wno-unused-value -Wno-unused-lambda-capture -Wno-uninitialized-const-reference -std=c++20 -verify %s

void* operator new(__SIZE_TYPE__, void*);

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type&& move(T&& t);
}
}

int foo(int x);
int bar(int* x);
int boo(int& x);
int far(const int& x);
int moved(int&& x);
int &ref(int x);

// Test self-references within initializers which are guaranteed to be
// uninitialized.
int a = a; // no-warning: used to signal intended lack of initialization.
int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
int c = (c + c); // expected-warning 2 {{variable 'c' is uninitialized when used within its own initialization}}
int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

// These don't warn as they don't require the value.
int g = sizeof(g);
void* ptr = &ptr;
int h = bar(&h);
int i = boo(i);
int j = far(j);
int k = __alignof__(k);

int l = k ? l : l;  // expected-warning 2{{variable 'l' is uninitialized when used within its own initialization}}
int m = 1 + (k ? m : m);  // expected-warning 2{{variable 'm' is uninitialized when used within its own initialization}}
int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
int o = std::move(o); // expected-warning {{variable 'o' is uninitialized when used within its own initialization}}
const int p = std::move(p); // expected-warning {{variable 'p' is uninitialized when used within its own initialization}}
int q = moved(std::move(q)); // expected-warning {{variable 'q' is uninitialized when used within its own initialization}}
int r = std::move((p ? q : (18, r))); // expected-warning {{variable 'r' is uninitialized when used within its own initialization}}
int s = r ?: s; // expected-warning {{variable 's' is uninitialized when used within its own initialization}}
int t = t ?: s; // expected-warning {{variable 't' is uninitialized when used within its own initialization}}
int u = (foo(u), s); // expected-warning {{variable 'u' is uninitialized when used within its own initialization}}
int v = (u += v); // expected-warning {{variable 'v' is uninitialized when used within its own initialization}}
int w = (w += 10); // expected-warning {{variable 'w' is uninitialized when used within its own initialization}}
int x = x++; // expected-warning {{variable 'x' is uninitialized when used within its own initialization}}
int y = ((s ? (y, v) : (77, y))++, sizeof(y)); // expected-warning {{variable 'y' is uninitialized when used within its own initialization}}
int z = ++ref(z); // expected-warning {{variable 'z' is uninitialized when used within its own initialization}}
int aa = (ref(aa) += 10); // expected-warning {{variable 'aa' is uninitialized when used within its own initialization}}
int bb = bb ? x : y; // expected-warning {{variable 'bb' is uninitialized when used within its own initialization}}

void test_stuff () {
  int a = a; // no-warning: used to signal intended lack of initialization.
  int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
  int c = (c + c); // expected-warning {{variable 'c' is uninitialized when used within its own initialization}}
  int d = ({ d + d ;}); // expected-warning {{variable 'd' is uninitialized when used within its own initialization}}
  int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
  int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

  // These don't warn as they don't require the value.
  int g = sizeof(g);
  void* ptr = &ptr;
  int h = bar(&h);
  int i = boo(i);
  int j = far(j);
  int k = __alignof__(k);

  int l = k ? l : l;  // expected-warning {{variable 'l' is uninitialized when used within its own initialization}}
  int m = 1 + (k ? m : m);  // expected-warning {{'m' is uninitialized when used within its own initialization}}
  int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
  int o = std::move(o);  // expected-warning {{variable 'o' is uninitialized when used within its own initialization}}
  const int p = std::move(p);  // expected-warning {{variable 'p' is uninitialized when used within its own initialization}}
  int q = moved(std::move(q));  // expected-warning {{variable 'q' is uninitialized when used within its own initialization}}
  int r = std::move((p ? q : (18, r))); // expected-warning {{variable 'r' is uninitialized when used within its own initialization}}
  int s = r ?: s; // expected-warning {{variable 's' is uninitialized when used within its own initialization}}
  int t = t ?: s; // expected-warning {{variable 't' is uninitialized when used within its own initialization}}
  int u = (foo(u), s); // expected-warning {{variable 'u' is uninitialized when used within its own initialization}}
  int v = (u += v); // expected-warning {{variable 'v' is uninitialized when used within its own initialization}}
  int w = (w += 10); // expected-warning {{variable 'w' is uninitialized when used within its own initialization}}
  int x = x++; // expected-warning {{variable 'x' is uninitialized when used within its own initialization}}
  int y = ((s ? (y, v) : (77, y))++, sizeof(y)); // expected-warning {{variable 'y' is uninitialized when used within its own initialization}}
  int z = ++ref(z);                              // expected-warning {{variable 'z' is uninitialized when used within its own initialization}}
  int aa = (ref(aa) += 10); // expected-warning {{variable 'aa' is uninitialized when used within its own initialization}}
  int bb = bb ? x : y; // expected-warning {{variable 'bb' is uninitialized when used within its own initialization}}

  for (;;) {
    int a = a; // no-warning: used to signal intended lack of initialization.
    int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
    int c = (c + c); // expected-warning {{variable 'c' is uninitialized when used within its own initialization}}
    int d = ({ d + d ;}); // expected-warning {{variable 'd' is uninitialized when used within its own initialization}}
    int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
    int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

    // These don't warn as they don't require the value.
    int g = sizeof(g);
    void* ptr = &ptr;
    int h = bar(&h);
    int i = boo(i);
    int j = far(j);
    int k = __alignof__(k);

    int l = k ? l : l;  // expected-warning {{variable 'l' is uninitialized when used within its own initialization}}
    int m = 1 + (k ? m : m);  // expected-warning {{'m' is uninitialized when used within its own initialization}}
    int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
    int o = std::move(o);  // expected-warning {{variable 'o' is uninitialized when used within its own initialization}}
    const int p = std::move(p);  // expected-warning {{variable 'p' is uninitialized when used within its own initialization}}
    int q = moved(std::move(q));  // expected-warning {{variable 'q' is uninitialized when used within its own initialization}}
    int r = std::move((p ? q : (18, r))); // expected-warning {{variable 'r' is uninitialized when used within its own initialization}}
    int s = r ?: s; // expected-warning {{variable 's' is uninitialized when used within its own initialization}}
    int t = t ?: s; // expected-warning {{variable 't' is uninitialized when used within its own initialization}}
    int u = (foo(u), s); // expected-warning {{variable 'u' is uninitialized when used within its own initialization}}
    int v = (u += v); // expected-warning {{variable 'v' is uninitialized when used within its own initialization}}
    int w = (w += 10); // expected-warning {{variable 'w' is uninitialized when used within its own initialization}}
    int x = x++; // expected-warning {{variable 'x' is uninitialized when used within its own initialization}}
    int y = ((s ? (y, v) : (77, y))++, sizeof(y)); // expected-warning {{variable 'y' is uninitialized when used within its own initialization}}
    int z = ++ref(z);                              // expected-warning {{variable 'z' is uninitialized when used within its own initialization}}
    int aa = (ref(aa) += 10); // expected-warning {{variable 'aa' is uninitialized when used within its own initialization}}
    int bb = bb ? x : y; // expected-warning {{variable 'bb' is uninitialized when used within its own initialization}}

  }
}

void test_comma() {
  int a;  // expected-note {{initialize the variable 'a' to silence this warning}}
  int b = (a, a ?: 2);  // expected-warning {{variable 'a' is uninitialized when used here}}
  int c = (a, a, b, c);  // expected-warning {{variable 'c' is uninitialized when used within its own initialization}}
  int d;  // expected-note {{initialize the variable 'd' to silence this warning}}
  int e = (foo(d), e, b); // expected-warning {{variable 'd' is uninitialized when used here}}
  int f;  // expected-note {{initialize the variable 'f' to silence this warning}}
  f = f + 1, 2;  // expected-warning {{variable 'f' is uninitialized when used here}}
  int h;
  int g = (h, g, 2);  // no-warning: h, g are evaluated but not used.
}

namespace member_ptr {
struct A {
  int x;
  int y;
  A(int x) : x{x} {}
};

void test_member_ptr() {
  int A::* px = &A::x;
  A a{a.*px}; // expected-warning {{variable 'a' is uninitialized when used within its own initialization}}
  A b = b; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
}
}

namespace const_ptr {
void foo(int *a);
void bar(const int *a);
void foobar(const int **a);

void test_const_ptr() {
  int a;
  int b;  // expected-note {{initialize the variable 'b' to silence this warning}}
  foo(&a);
  bar(&b);
  b = a + b; // expected-warning {{variable 'b' is uninitialized when used here}}
  int *ptr;  //expected-note {{initialize the variable 'ptr' to silence this warning}}
  const int *ptr2;
  foo(ptr); // expected-warning {{variable 'ptr' is uninitialized when used here}}
  foobar(&ptr2);
}
}

// Also test similar constructs in a field's initializer.
struct S {
  int x;
  int y;
  const int z = 5;
  void *ptr;

  S(bool (*)[1]) : x(x) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(bool (*)[2]) : x(x + 1) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(bool (*)[3]) : x(x + x) {} // expected-warning 2{{field 'x' is uninitialized when used here}}
  S(bool (*)[4]) : x(static_cast<long>(x) + 1) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(bool (*)[5]) : x(foo(x)) {} // expected-warning {{field 'x' is uninitialized when used here}}

  // These don't actually require the value of x and so shouldn't warn.
  S(char (*)[1]) : x(sizeof(x)) {}
  S(char (*)[2]) : ptr(&ptr) {}
  S(char (*)[3]) : x(bar(&x)) {}
  S(char (*)[4]) : x(boo(x)) {}
  S(char (*)[5]) : x(far(x)) {}
  S(char (*)[6]) : x(__alignof__(x)) {}

  S(int (*)[1]) : x(0), y(x ? y : y) {} // expected-warning 2{{field 'y' is uninitialized when used here}}
  S(int (*)[2]) : x(0), y(1 + (x ? y : y)) {} // expected-warning 2{{field 'y' is uninitialized when used here}}
  S(int (*)[3]) : x(-x) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[4]) : x(std::move(x)) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[5]) : z(std::move(z)) {} // expected-warning {{field 'z' is uninitialized when used here}}
  S(int (*)[6]) : x(moved(std::move(x))) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[7]) : x(0), y(std::move((x ? x : (18, y)))) {} // expected-warning {{field 'y' is uninitialized when used here}}
  S(int (*)[8]) : x(0), y(x ?: y) {} // expected-warning {{field 'y' is uninitialized when used here}}
  S(int (*)[9]) : x(0), y(y ?: x) {} // expected-warning {{field 'y' is uninitialized when used here}}
  S(int (*)[10]) : x(0), y((foo(y), x)) {} // expected-warning {{field 'y' is uninitialized when used here}}
  S(int (*)[11]) : x(0), y(x += y) {} // expected-warning {{field 'y' is uninitialized when used here}}
  S(int (*)[12]) : x(x += 10) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[13]) : x(x++) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[14]) : x(0), y(((x ? (y, x) : (77, y))++, sizeof(y))) {} // expected-warning {{field 'y' is uninitialized when used here}}
  S(int (*)[15]) : x(++ref(x)) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[16]) : x((ref(x) += 10)) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(int (*)[17]) : x(0), y(y ? x : x) {} // expected-warning {{field 'y' is uninitialized when used here}}
};

// Test self-references with record types.
class A {
  // Non-POD class.
  public:
    enum count { ONE, TWO, THREE };
    int num;
    static int count;
    int get() const { return num; }
    int get2() { return num; }
    int set(int x) { num = x; return num; }
    static int zero() { return 0; }

    A() {}
    A(A const &a) {}
    A(int x) {}
    A(int *x) {}
    A(A *a) {}
    A(A &&a) {}
    ~A();
    bool operator!();
    bool operator!=(const A&);
};

bool operator!=(int, const A&);

A getA() { return A(); }
A getA(int x) { return A(); }
A getA(A* a) { return A(); }
A getA(A a) { return A(); }
A moveA(A&& a) { return A(); }
A const_refA(const A& a) { return A(); }

void setupA(bool x) {
  A a1;
  a1.set(a1.get());
  A a2(a1.get());
  A a3(a1);
  A a4(&a4);
  A a5(a5.zero());
  A a6(a6.ONE);
  A a7 = getA();
  A a8 = getA(a8.TWO);
  A a9 = getA(&a9);
  A a10(a10.count);

  A a11(a11);  // expected-warning {{variable 'a11' is uninitialized when used within its own initialization}}
  A a12(a12.get());  // expected-warning {{variable 'a12' is uninitialized when used within its own initialization}}
  A a13(a13.num);  // expected-warning {{variable 'a13' is uninitialized when used within its own initialization}}
  A a14 = A(a14);  // expected-warning {{variable 'a14' is uninitialized when used within its own initialization}}
  A a15 = getA(a15.num);  // expected-warning {{variable 'a15' is uninitialized when used within its own initialization}}
  A a16(&a16.num);  // expected-warning {{variable 'a16' is uninitialized when used within its own initialization}}
  A a17(a17.get2());  // expected-warning {{variable 'a17' is uninitialized when used within its own initialization}}
  A a18 = x ? a18 : a17;  // expected-warning {{variable 'a18' is uninitialized when used within its own initialization}}
  A a19 = getA(x ? a19 : a17);  // expected-warning {{variable 'a19' is uninitialized when used within its own initialization}}
  A a20{a20};  // expected-warning {{variable 'a20' is uninitialized when used within its own initialization}}
  A a21 = {a21};  // expected-warning {{variable 'a21' is uninitialized when used within its own initialization}}

  // FIXME: Make the local uninitialized warning consistent with the global
  // uninitialized checking.
  A *a22 = new A(a22->count);  // expected-warning {{variable 'a22' is uninitialized when used within its own initialization}}
  A *a23 = new A(a23->ONE);  // expected-warning {{variable 'a23' is uninitialized when used within its own initialization}}
  A *a24 = new A(a24->TWO);  // expected-warning {{variable 'a24' is uninitialized when used within its own initialization}}
  A *a25 = new A(a25->zero());  // expected-warning {{variable 'a25' is uninitialized when used within its own initialization}}

  A *a26 = new A(a26->get());    // expected-warning {{variable 'a26' is uninitialized when used within its own initialization}}
  A *a27 = new A(a27->get2());  // expected-warning {{variable 'a27' is uninitialized when used within its own initialization}}
  A *a28 = new A(a28->num);  // expected-warning {{variable 'a28' is uninitialized when used within its own initialization}}

  const A a29(a29);  // expected-warning {{variable 'a29' is uninitialized when used within its own initialization}}
  const A a30 = a30;  // expected-warning {{variable 'a30' is uninitialized when used within its own initialization}}

  A a31 = std::move(a31);  // expected-warning {{variable 'a31' is uninitialized when used within its own initialization}}
  A a32 = moveA(std::move(a32));  // expected-warning {{variable 'a32' is uninitialized when used within its own initialization}}
  A a33 = A(std::move(a33));   // expected-warning {{variable 'a33' is uninitialized when used within its own initialization}}
  A a34(std::move(a34));   // expected-warning {{variable 'a34' is uninitialized when used within its own initialization}}
  A a35 = std::move(x ? a34 : (37, a35));  // expected-warning {{variable 'a35' is uninitialized when used within its own initialization}}

  A a36 = const_refA(a36);
  A a37(const_refA(a37));

  A a38({a38});  // expected-warning {{variable 'a38' is uninitialized when used within its own initialization}}
  A a39 = {a39};  // expected-warning {{variable 'a39' is uninitialized when used within its own initialization}}
  A a40 = A({a40});  // expected-warning {{variable 'a40' is uninitialized when used within its own initialization}}

  A a41 = !a41;  // expected-warning {{variable 'a41' is uninitialized when used within its own initialization}}
  A a42 = !(a42);  // expected-warning {{variable 'a42' is uninitialized when used within its own initialization}}
  A a43 = a43 != a42;  // expected-warning {{variable 'a43' is uninitialized when used within its own initialization}}
  A a44 = a43 != a44;  // expected-warning {{variable 'a44' is uninitialized when used within its own initialization}}
  A a45 = a45 != a45;  // expected-warning 2{{variable 'a45' is uninitialized when used within its own initialization}}
  A a46 = 0 != a46;  // expected-warning {{variable 'a46' is uninitialized when used within its own initialization}}

  A a47(a47.set(a47.num));  // expected-warning 2{{variable 'a47' is uninitialized when used within its own initialization}}
  A a48(a47.set(a48.num));  // expected-warning {{variable 'a48' is uninitialized when used within its own initialization}}
  A a49(a47.set(a48.num));
}

bool cond;

A a1;
A a2(a1.get());
A a3(a1);
A a4(&a4);
A a5(a5.zero());
A a6(a6.ONE);
A a7 = getA();
A a8 = getA(a8.TWO);
A a9 = getA(&a9);
A a10(a10.count);

A a11(a11);  // expected-warning {{variable 'a11' is uninitialized when used within its own initialization}}
A a12(a12.get());  // expected-warning {{variable 'a12' is uninitialized when used within its own initialization}}
A a13(a13.num);  // expected-warning {{variable 'a13' is uninitialized when used within its own initialization}}
A a14 = A(a14);  // expected-warning {{variable 'a14' is uninitialized when used within its own initialization}}
A a15 = getA(a15.num);  // expected-warning {{variable 'a15' is uninitialized when used within its own initialization}}
A a16(&a16.num);  // expected-warning {{variable 'a16' is uninitialized when used within its own initialization}}
A a17(a17.get2());  // expected-warning {{variable 'a17' is uninitialized when used within its own initialization}}
A a18 = cond ? a18 : a17;  // expected-warning {{variable 'a18' is uninitialized when used within its own initialization}}
A a19 = getA(cond ? a19 : a17);  // expected-warning {{variable 'a19' is uninitialized when used within its own initialization}}
A a20{a20};  // expected-warning {{variable 'a20' is uninitialized when used within its own initialization}}
A a21 = {a21};  // expected-warning {{variable 'a21' is uninitialized when used within its own initialization}}

A *a22 = new A(a22->count);
A *a23 = new A(a23->ONE);
A *a24 = new A(a24->TWO);
A *a25 = new A(a25->zero());

A *a26 = new A(a26->get());    // expected-warning {{variable 'a26' is uninitialized when used within its own initialization}}
A *a27 = new A(a27->get2());  // expected-warning {{variable 'a27' is uninitialized when used within its own initialization}}
A *a28 = new A(a28->num);  // expected-warning {{variable 'a28' is uninitialized when used within its own initialization}}

const A a29(a29);  // expected-warning {{variable 'a29' is uninitialized when used within its own initialization}}
const A a30 = a30;  // expected-warning {{variable 'a30' is uninitialized when used within its own initialization}}

A a31 = std::move(a31);  // expected-warning {{variable 'a31' is uninitialized when used within its own initialization}}
A a32 = moveA(std::move(a32));  // expected-warning {{variable 'a32' is uninitialized when used within its own initialization}}
A a33 = A(std::move(a33));   // expected-warning {{variable 'a33' is uninitialized when used within its own initialization}}
A a34(std::move(a34));   // expected-warning {{variable 'a34' is uninitialized when used within its own initialization}}
A a35 = std::move(x ? a34 : (37, a35));  // expected-warning {{variable 'a35' is uninitialized when used within its own initialization}}

A a36 = const_refA(a36);
A a37(const_refA(a37));

A a38({a38});  // expected-warning {{variable 'a38' is uninitialized when used within its own initialization}}
A a39 = {a39};  // expected-warning {{variable 'a39' is uninitialized when used within its own initialization}}
A a40 = A({a40});  // expected-warning {{variable 'a40' is uninitialized when used within its own initialization}}

A a41 = !a41;  // expected-warning {{variable 'a41' is uninitialized when used within its own initialization}}
A a42 = !(a42);  // expected-warning {{variable 'a42' is uninitialized when used within its own initialization}}
A a43 = a43 != a42;  // expected-warning {{variable 'a43' is uninitialized when used within its own initialization}}
A a44 = a43 != a44;  // expected-warning {{variable 'a44' is uninitialized when used within its own initialization}}
A a45 = a45 != a45;  // expected-warning 2{{variable 'a45' is uninitialized when used within its own initialization}}

A a46 = 0 != a46;  // expected-warning {{variable 'a46' is uninitialized when used within its own initialization}}

A a47(a47.set(a47.num));  // expected-warning 2{{variable 'a47' is uninitialized when used within its own initialization}}
A a48(a47.set(a48.num));  // expected-warning {{variable 'a48' is uninitialized when used within its own initialization}}
A a49(a47.set(a48.num));

class T {
  A a, a2;
  const A c_a;
  A* ptr_a;

  T() {}
  T(bool (*)[1]) : a() {}
  T(bool (*)[2]) : a2(a.get()) {}
  T(bool (*)[3]) : a2(a) {}
  T(bool (*)[4]) : a(&a) {}
  T(bool (*)[5]) : a(a.zero()) {}
  T(bool (*)[6]) : a(a.ONE) {}
  T(bool (*)[7]) : a(getA()) {}
  T(bool (*)[8]) : a2(getA(a.TWO)) {}
  T(bool (*)[9]) : a(getA(&a)) {}
  T(bool (*)[10]) : a(a.count) {}

  T(bool (*)[11]) : a(a) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[12]) : a(a.get()) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[13]) : a(a.num) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[14]) : a(A(a)) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[15]) : a(getA(a.num)) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[16]) : a(&a.num) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[17]) : a(a.get2()) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[18]) : a2(cond ? a2 : a) {}  // expected-warning {{field 'a2' is uninitialized when used here}}
  T(bool (*)[19]) : a2(cond ? a2 : a) {}  // expected-warning {{field 'a2' is uninitialized when used here}}
  T(bool (*)[20]) : a{a} {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[21]) : a({a}) {}  // expected-warning {{field 'a' is uninitialized when used here}}

  T(bool (*)[22]) : ptr_a(new A(ptr_a->count)) {}
  T(bool (*)[23]) : ptr_a(new A(ptr_a->ONE)) {}
  T(bool (*)[24]) : ptr_a(new A(ptr_a->TWO)) {}
  T(bool (*)[25]) : ptr_a(new A(ptr_a->zero())) {}

  T(bool (*)[26]) : ptr_a(new A(ptr_a->get())) {}  // expected-warning {{field 'ptr_a' is uninitialized when used here}}
  T(bool (*)[27]) : ptr_a(new A(ptr_a->get2())) {}  // expected-warning {{field 'ptr_a' is uninitialized when used here}}
  T(bool (*)[28]) : ptr_a(new A(ptr_a->num)) {}  // expected-warning {{field 'ptr_a' is uninitialized when used here}}

  T(bool (*)[29]) : c_a(c_a) {}  // expected-warning {{field 'c_a' is uninitialized when used here}}
  T(bool (*)[30]) : c_a(A(c_a)) {}  // expected-warning {{field 'c_a' is uninitialized when used here}}

  T(bool (*)[31]) : a(std::move(a)) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[32]) : a(moveA(std::move(a))) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[33]) : a(A(std::move(a))) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[34]) : a(A(std::move(a))) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[35]) : a2(std::move(x ? a : (37, a2))) {}  // expected-warning {{field 'a2' is uninitialized when used here}}

  T(bool (*)[36]) : a(const_refA(a)) {}
  T(bool (*)[37]) : a(A(const_refA(a))) {}

  T(bool (*)[38]) : a({a}) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[39]) : a{a} {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[40]) : a({a}) {}  // expected-warning {{field 'a' is uninitialized when used here}}

  T(bool (*)[41]) : a(!a) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[42]) : a(!(a)) {}  // expected-warning {{field 'a' is uninitialized when used here}}
  T(bool (*)[43]) : a(), a2(a2 != a) {}  // expected-warning {{field 'a2' is uninitialized when used here}}
  T(bool (*)[44]) : a(), a2(a != a2) {}  // expected-warning {{field 'a2' is uninitialized when used here}}
  T(bool (*)[45]) : a(a != a) {}  // expected-warning 2{{field 'a' is uninitialized when used here}}
  T(bool (*)[46]) : a(0 != a) {}  // expected-warning {{field 'a' is uninitialized when used here}}

  T(bool (*)[47]) : a2(a2.set(a2.num)) {}  // expected-warning 2{{field 'a2' is uninitialized when used here}}
  T(bool (*)[48]) : a2(a.set(a2.num)) {}  // expected-warning {{field 'a2' is uninitialized when used here}}
  T(bool (*)[49]) : a2(a.set(a.num)) {}

};

struct B {
  // POD struct.
  int x;
  int *y;
};

B getB() { return B(); };
B getB(int x) { return B(); };
B getB(int *x) { return B(); };
B getB(B *b) { return B(); };
B moveB(B &&b) { return B(); };

B* getPtrB() { return 0; };
B* getPtrB(int x) { return 0; };
B* getPtrB(int *x) { return 0; };
B* getPtrB(B **b) { return 0; };

void setupB(bool x) {
  B b1;
  B b2(b1);
  B b3 = { 5, &b3.x };
  B b4 = getB();
  B b5 = getB(&b5);
  B b6 = getB(&b6.x);

  // Silence unused warning
  (void) b2;
  (void) b4;

  B b7(b7);  // expected-warning {{variable 'b7' is uninitialized when used within its own initialization}}
  B b8 = getB(b8.x);  // expected-warning {{variable 'b8' is uninitialized when used within its own initialization}}
  B b9 = getB(b9.y);  // expected-warning {{variable 'b9' is uninitialized when used within its own initialization}}
  B b10 = getB(-b10.x);  // expected-warning {{variable 'b10' is uninitialized when used within its own initialization}}

  B* b11 = 0;
  B* b12(b11);
  B* b13 = getPtrB();
  B* b14 = getPtrB(&b14);

  (void) b12;
  (void) b13;

  B* b15 = getPtrB(b15->x);  // expected-warning {{variable 'b15' is uninitialized when used within its own initialization}}
  B* b16 = getPtrB(b16->y);  // expected-warning {{variable 'b16' is uninitialized when used within its own initialization}}

  B b17 = { b17.x = 5, b17.y = 0 };
  B b18 = { b18.x + 1, b18.y };  // expected-warning 2{{variable 'b18' is uninitialized when used within its own initialization}}

  const B b19 = b19;  // expected-warning {{variable 'b19' is uninitialized when used within its own initialization}}
  const B b20(b20);  // expected-warning {{variable 'b20' is uninitialized when used within its own initialization}}

  B b21 = std::move(b21);  // expected-warning {{variable 'b21' is uninitialized when used within its own initialization}}
  B b22 = moveB(std::move(b22));  // expected-warning {{variable 'b22' is uninitialized when used within its own initialization}}
  B b23 = B(std::move(b23));   // expected-warning {{variable 'b23' is uninitialized when used within its own initialization}}
  B b24 = std::move(x ? b23 : (18, b24));  // expected-warning {{variable 'b24' is uninitialized when used within its own initialization}}
}

B b1;
B b2(b1);
B b3 = { 5, &b3.x };
B b4 = getB();
B b5 = getB(&b5);
B b6 = getB(&b6.x);

B b7(b7);  // expected-warning {{variable 'b7' is uninitialized when used within its own initialization}}
B b8 = getB(b8.x);  // expected-warning {{variable 'b8' is uninitialized when used within its own initialization}}
B b9 = getB(b9.y);  // expected-warning {{variable 'b9' is uninitialized when used within its own initialization}}
B b10 = getB(-b10.x);  // expected-warning {{variable 'b10' is uninitialized when used within its own initialization}}

B* b11 = 0;
B* b12(b11);
B* b13 = getPtrB();
B* b14 = getPtrB(&b14);

B* b15 = getPtrB(b15->x);  // expected-warning {{variable 'b15' is uninitialized when used within its own initialization}}
B* b16 = getPtrB(b16->y);  // expected-warning {{variable 'b16' is uninitialized when used within its own initialization}}

B b17 = { b17.x = 5, b17.y = 0 };
B b18 = { b18.x + 1, b18.y };  // expected-warning 2{{variable 'b18' is uninitialized when used within its own initialization}}

const B b19 = b19;  // expected-warning {{variable 'b19' is uninitialized when used within its own initialization}}
const B b20(b20);  // expected-warning {{variable 'b20' is uninitialized when used within its own initialization}}

B b21 = std::move(b21);  // expected-warning {{variable 'b21' is uninitialized when used within its own initialization}}
B b22 = moveB(std::move(b22));  // expected-warning {{variable 'b22' is uninitialized when used within its own initialization}}
B b23 = B(std::move(b23));   // expected-warning {{variable 'b23' is uninitialized when used within its own initialization}}
B b24 = std::move(x ? b23 : (18, b24));  // expected-warning {{variable 'b24' is uninitialized when used within its own initialization}}

class U {
  B b1, b2;
  B *ptr1, *ptr2;
  const B constb = {};

  U() {}
  U(bool (*)[1]) : b1() {}
  U(bool (*)[2]) : b2(b1) {}
  U(bool (*)[3]) : b1{ 5, &b1.x } {}
  U(bool (*)[4]) : b1(getB()) {}
  U(bool (*)[5]) : b1(getB(&b1)) {}
  U(bool (*)[6]) : b1(getB(&b1.x)) {}

  U(bool (*)[7]) : b1(b1) {}  // expected-warning {{field 'b1' is uninitialized when used here}}
  U(bool (*)[8]) : b1(getB(b1.x)) {}  // expected-warning {{field 'b1' is uninitialized when used here}}
  U(bool (*)[9]) : b1(getB(b1.y)) {}  // expected-warning {{field 'b1' is uninitialized when used here}}
  U(bool (*)[10]) : b1(getB(-b1.x)) {}  // expected-warning {{field 'b1' is uninitialized when used here}}

  U(bool (*)[11]) : ptr1(0) {}
  U(bool (*)[12]) : ptr1(0), ptr2(ptr1) {}
  U(bool (*)[13]) : ptr1(getPtrB()) {}
  U(bool (*)[14]) : ptr1(getPtrB(&ptr1)) {}

  U(bool (*)[15]) : ptr1(getPtrB(ptr1->x)) {}  // expected-warning {{field 'ptr1' is uninitialized when used here}}
  U(bool (*)[16]) : ptr2(getPtrB(ptr2->y)) {}  // expected-warning {{field 'ptr2' is uninitialized when used here}}

  U(bool (*)[17]) : b1 { b1.x = 5, b1.y = 0 } {}
  U(bool (*)[18]) : b1 { b1.x + 1, b1.y } {}  // expected-warning 2{{field 'b1' is uninitialized when used here}}

  U(bool (*)[19]) : constb(constb) {}  // expected-warning {{field 'constb' is uninitialized when used here}}
  U(bool (*)[20]) : constb(B(constb)) {}  // expected-warning {{field 'constb' is uninitialized when used here}}

  U(bool (*)[21]) : b1(std::move(b1)) {}  // expected-warning {{field 'b1' is uninitialized when used here}}
  U(bool (*)[22]) : b1(moveB(std::move(b1))) {}  // expected-warning {{field 'b1' is uninitialized when used here}}
  U(bool (*)[23]) : b1(B(std::move(b1))) {}  // expected-warning {{field 'b1' is uninitialized when used here}}
  U(bool (*)[24]) : b2(std::move(x ? b1 : (18, b2))) {}  // expected-warning {{field 'b2' is uninitialized when used here}}
};

struct C { char a[100], *e; } car = { .e = car.a };

namespace rdar10398199 {
  class FooBase { protected: ~FooBase() {} };
  class Foo : public FooBase {
  public:
    operator int&() const;
  };
  void stuff();
  template <typename T> class FooImpl : public Foo {
    T val;
  public:
    FooImpl(const T &x) : val(x) {}
    ~FooImpl() { stuff(); }
  };

  template <typename T> FooImpl<T> makeFoo(const T& x) {
    return FooImpl<T>(x);
  }

  void test() {
    const Foo &x = makeFoo(42);
    const int&y = makeFoo(42u);
    (void)x;
    (void)y;
  };
}

// PR 12325 - this was a false uninitialized value warning due to
// a broken CFG.
int pr12325(int params) {
  int x = ({
    while (false)
      ;
    int _v = params;
    if (false)
      ;
    _v; // no-warning
  });
  return x;
}

// Test lambda expressions with -Wuninitialized
int test_lambda() {
  auto f1 = [] (int x, int y) { int z; return x + y + z; }; // expected-warning{{variable 'z' is uninitialized when used here}} expected-note {{initialize the variable 'z' to silence this warning}}
  return f1(1, 2);
}

namespace {
  struct A {
    enum { A1 };
    static int A2() {return 5;}
    int A3;
    int A4() { return 5;}
  };

  struct B {
    A a;
  };

  struct C {
    C() {}
    C(int x) {}
    static A a;
    B b;
  };
  A C::a = A();

  // Accessing non-static members will give a warning.
  struct D {
    C c;
    D(char (*)[1]) : c(c.b.a.A1) {}
    D(char (*)[2]) : c(c.b.a.A2()) {}
    D(char (*)[3]) : c(c.b.a.A3) {}    // expected-warning {{field 'c' is uninitialized when used here}}
    D(char (*)[4]) : c(c.b.a.A4()) {}  // expected-warning {{field 'c' is uninitialized when used here}}

    // c::a is static, so it is already initialized
    D(char (*)[5]) : c(c.a.A1) {}
    D(char (*)[6]) : c(c.a.A2()) {}
    D(char (*)[7]) : c(c.a.A3) {}
    D(char (*)[8]) : c(c.a.A4()) {}
  };

  struct E {
    int b = 1;
    int c = 1;
    int a;  // This field needs to be last to prevent the cross field
            // uninitialized warning.
    E(char (*)[1]) : a(a ? b : c) {}  // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[2]) : a(b ? a : a) {} // expected-warning 2{{field 'a' is uninitialized when used here}}
    E(char (*)[3]) : a(b ? (a) : c) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[4]) : a(b ? c : (a+c)) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[5]) : a(b ? c : b) {}

    E(char (*)[6]) : a(a ?: a) {} // expected-warning 2{{field 'a' is uninitialized when used here}}
    E(char (*)[7]) : a(b ?: a) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[8]) : a(a ?: c) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[9]) : a(b ?: c) {}

    E(char (*)[10]) : a((a, a, b)) {}
    E(char (*)[11]) : a((c + a, a + 1, b)) {} // expected-warning 2{{field 'a' is uninitialized when used here}}
    E(char (*)[12]) : a((b + c, c, a)) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[13]) : a((a, a, a, a)) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[14]) : a((b, c, c)) {}
    E(char (*)[15]) : a(b ?: a) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[16]) : a(a ?: b) {} // expected-warning {{field 'a' is uninitialized when used here}}
  };

  struct F {
    int a;
    F* f;
    F(int) {}
    F() {}
  };

  int F::*ptr = &F::a;
  F* F::*f_ptr = &F::f;
  struct G {
    F f1, f2;
    F *f3, *f4;
    G(char (*)[1]) : f1(f1) {} // expected-warning {{field 'f1' is uninitialized when used here}}
    G(char (*)[2]) : f2(f1) {}
    G(char (*)[3]) : f2(F()) {}

    G(char (*)[4]) : f1(f1.*ptr) {} // expected-warning {{field 'f1' is uninitialized when used here}}
    G(char (*)[5]) : f2(f1.*ptr) {}

    G(char (*)[6]) : f3(f3) {}  // expected-warning {{field 'f3' is uninitialized when used here}}
    G(char (*)[7]) : f3(f3->*f_ptr) {} // expected-warning {{field 'f3' is uninitialized when used here}}
    G(char (*)[8]) : f3(new F(f3->*ptr)) {} // expected-warning {{field 'f3' is uninitialized when used here}}
  };

  struct H {
    H() : a(a) {}  // expected-warning {{field 'a' is uninitialized when used here}}
    const A a;
  };
}

namespace statics {
  static int a = a; // no-warning: used to signal intended lack of initialization.
  static int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
  static int c = (c + c); // expected-warning 2{{variable 'c' is uninitialized when used within its own initialization}}
  static int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
  static int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

  // These don't warn as they don't require the value.
  static int g = sizeof(g);
  int gg = g;  // Silence unneeded warning
  static void* ptr = &ptr;
  static int h = bar(&h);
  static int i = boo(i);
  static int j = far(j);
  static int k = __alignof__(k);

  static int l = k ? l : l;  // expected-warning 2{{variable 'l' is uninitialized when used within its own initialization}}
  static int m = 1 + (k ? m : m);  // expected-warning 2{{variable 'm' is uninitialized when used within its own initialization}}
  static int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
  static int o = std::move(o); // expected-warning {{variable 'o' is uninitialized when used within its own initialization}}
  static const int p = std::move(p); // expected-warning {{variable 'p' is uninitialized when used within its own initialization}}
  static int q = moved(std::move(q)); // expected-warning {{variable 'q' is uninitialized when used within its own initialization}}
  static int r = std::move((p ? q : (18, r))); // expected-warning {{variable 'r' is uninitialized when used within its own initialization}}
  static int s = r ?: s; // expected-warning {{variable 's' is uninitialized when used within its own initialization}}
  static int t = t ?: s; // expected-warning {{variable 't' is uninitialized when used within its own initialization}}
  static int u = (foo(u), s); // expected-warning {{variable 'u' is uninitialized when used within its own initialization}}
  static int v = (u += v); // expected-warning {{variable 'v' is uninitialized when used within its own initialization}}
  static int w = (w += 10); // expected-warning {{variable 'w' is uninitialized when used within its own initialization}}
  static int x = x++; // expected-warning {{variable 'x' is uninitialized when used within its own initialization}}
  static int y = ((s ? (y, v) : (77, y))++, sizeof(y)); // expected-warning {{variable 'y' is uninitialized when used within its own initialization}}
  static int z = ++ref(z); // expected-warning {{variable 'z' is uninitialized when used within its own initialization}}
  static int aa = (ref(aa) += 10); // expected-warning {{variable 'aa' is uninitialized when used within its own initialization}}
  static int bb = bb ? x : y; // expected-warning {{variable 'bb' is uninitialized when used within its own initialization}}


  void test() {
    static int a = a; // no-warning: used to signal intended lack of initialization.
    static int b = b + 1; // expected-warning {{static variable 'b' is suspiciously used within its own initialization}}
    static int c = (c + c); // expected-warning 2{{static variable 'c' is suspiciously used within its own initialization}}
    static int d = ({ d + d ;}); // expected-warning 2{{static variable 'd' is suspiciously used within its own initialization}}
    static int e = static_cast<long>(e) + 1; // expected-warning {{static variable 'e' is suspiciously used within its own initialization}}
    static int f = foo(f); // expected-warning {{static variable 'f' is suspiciously used within its own initialization}}

    // These don't warn as they don't require the value.
    static int g = sizeof(g);
    static void* ptr = &ptr;
    static int h = bar(&h);
    static int i = boo(i);
    static int j = far(j);
    static int k = __alignof__(k);

    static int l = k ? l : l;  // expected-warning 2{{static variable 'l' is suspiciously used within its own initialization}}
    static int m = 1 + (k ? m : m);  // expected-warning 2{{static variable 'm' is suspiciously used within its own initialization}}
    static int n = -n;  // expected-warning {{static variable 'n' is suspiciously used within its own initialization}}
    static int o = std::move(o);  // expected-warning {{static variable 'o' is suspiciously used within its own initialization}}
    static const int p = std::move(p);  // expected-warning {{static variable 'p' is suspiciously used within its own initialization}}
    static int q = moved(std::move(q));  // expected-warning {{static variable 'q' is suspiciously used within its own initialization}}
    static int r = std::move((p ? q : (18, r)));  // expected-warning {{static variable 'r' is suspiciously used within its own initialization}}
    static int s = r ?: s;  // expected-warning {{static variable 's' is suspiciously used within its own initialization}}
    static int t = t ?: s;  // expected-warning {{static variable 't' is suspiciously used within its own initialization}}
    static int u = (foo(u), s);  // expected-warning {{static variable 'u' is suspiciously used within its own initialization}}
    static int v = (u += v);  // expected-warning {{static variable 'v' is suspiciously used within its own initialization}}
    static int w = (w += 10);  // expected-warning {{static variable 'w' is suspiciously used within its own initialization}}
    static int x = x++;  // expected-warning {{static variable 'x' is suspiciously used within its own initialization}}
    static int y = ((s ? (y, v) : (77, y))++, sizeof(y));  // expected-warning {{static variable 'y' is suspiciously used within its own initialization}}
    static int z = ++ref(z); // expected-warning {{static variable 'z' is suspiciously used within its own initialization}}
    static int aa = (ref(aa) += 10); // expected-warning {{static variable 'aa' is suspiciously used within its own initialization}}
    static int bb = bb ? x : y; // expected-warning {{static variable 'bb' is suspiciously used within its own initialization}}

    for (;;) {
      static int a = a; // no-warning: used to signal intended lack of initialization.
      static int b = b + 1; // expected-warning {{static variable 'b' is suspiciously used within its own initialization}}
      static int c = (c + c); // expected-warning 2{{static variable 'c' is suspiciously used within its own initialization}}
      static int d = ({ d + d ;}); // expected-warning 2{{static variable 'd' is suspiciously used within its own initialization}}
      static int e = static_cast<long>(e) + 1; // expected-warning {{static variable 'e' is suspiciously used within its own initialization}}
      static int f = foo(f); // expected-warning {{static variable 'f' is suspiciously used within its own initialization}}

      // These don't warn as they don't require the value.
      static int g = sizeof(g);
      static void* ptr = &ptr;
      static int h = bar(&h);
      static int i = boo(i);
      static int j = far(j);
      static int k = __alignof__(k);

      static int l = k ? l : l;  // expected-warning 2{{static variable 'l' is suspiciously used within its own initialization}}
      static int m = 1 + (k ? m : m); // expected-warning 2{{static variable 'm' is suspiciously used within its own initialization}}
      static int n = -n;  // expected-warning {{static variable 'n' is suspiciously used within its own initialization}}
      static int o = std::move(o);  // expected-warning {{static variable 'o' is suspiciously used within its own initialization}}
      static const int p = std::move(p);  // expected-warning {{static variable 'p' is suspiciously used within its own initialization}}
      static int q = moved(std::move(q));  // expected-warning {{static variable 'q' is suspiciously used within its own initialization}}
      static int r = std::move((p ? q : (18, r)));  // expected-warning {{static variable 'r' is suspiciously used within its own initialization}}
      static int s = r ?: s;  // expected-warning {{static variable 's' is suspiciously used within its own initialization}}
      static int t = t ?: s;  // expected-warning {{static variable 't' is suspiciously used within its own initialization}}
      static int u = (foo(u), s);  // expected-warning {{static variable 'u' is suspiciously used within its own initialization}}
      static int v = (u += v);  // expected-warning {{static variable 'v' is suspiciously used within its own initialization}}
      static int w = (w += 10);  // expected-warning {{static variable 'w' is suspiciously used within its own initialization}}
      static int x = x++;  // expected-warning {{static variable 'x' is suspiciously used within its own initialization}}
      static int y = ((s ? (y, v) : (77, y))++, sizeof(y));  // expected-warning {{static variable 'y' is suspiciously used within its own initialization}}
      static int z = ++ref(z); // expected-warning {{static variable 'z' is suspiciously used within its own initialization}}
      static int aa = (ref(aa) += 10); // expected-warning {{static variable 'aa' is suspiciously used within its own initialization}}
      static int bb = bb ? x : y; // expected-warning {{static variable 'bb' is suspiciously used within its own initialization}}
    }
  }
}

namespace in_class_initializers {
  struct S {
    S() : a(a + 1) {} // expected-warning{{field 'a' is uninitialized when used here}}
    int a = 42; // Note: because a is in a member initializer list, this initialization is ignored.
  };

  struct T {
    T() : b(a + 1) {} // No-warning.
    int a = 42;
    int b;
  };

  struct U {
    U() : a(b + 1), b(a + 1) {} // expected-warning{{field 'b' is uninitialized when used here}}
    int a = 42; // Note: because a and b are in the member initializer list, these initializers are ignored.
    int b = 1;
  };
}

namespace references {
  int &a = a; // expected-warning{{reference 'a' is not yet bound to a value when used within its own initialization}}
  int &b(b); // expected-warning{{reference 'b' is not yet bound to a value when used within its own initialization}}
  int &c = a ? b : c; // expected-warning{{reference 'c' is not yet bound to a value when used within its own initialization}}
  int &d{d}; // expected-warning{{reference 'd' is not yet bound to a value when used within its own initialization}}
  int &e = d ?: e; // expected-warning{{reference 'e' is not yet bound to a value when used within its own initialization}}
  int &f = f ?: d; // expected-warning{{reference 'f' is not yet bound to a value when used within its own initialization}}

  int &return_ref1(int);
  int &return_ref2(int&);

  int &g = return_ref1(g); // expected-warning{{reference 'g' is not yet bound to a value when used within its own initialization}}
  int &h = return_ref2(h); // expected-warning{{reference 'h' is not yet bound to a value when used within its own initialization}}

  struct S {
    S() : a(a) {} // expected-warning{{reference 'a' is not yet bound to a value when used here}}
    int &a;
  };

  void test() {
    int &a = a; // expected-warning{{reference 'a' is not yet bound to a value when used within its own initialization}}
    int &b(b); // expected-warning{{reference 'b' is not yet bound to a value when used within its own initialization}}
    int &c = a ? b : c; // expected-warning{{reference 'c' is not yet bound to a value when used within its own initialization}}
    int &d{d}; // expected-warning{{reference 'd' is not yet bound to a value when used within its own initialization}}
  }

  struct T {
    T() // expected-note{{during field initialization in this constructor}}
     : a(b), b(a) {} // expected-warning{{reference 'b' is not yet bound to a value when used here}}
    int &a, &b;
    int &c = c; // expected-warning{{reference 'c' is not yet bound to a value when used here}}
  };

  int x;
  struct U {
    U() : b(a) {} // No-warning.
    int &a = x;
    int &b;
  };
}

namespace operators {
  struct A {
    A(bool);
    bool operator==(A);
  };

  A makeA();

  A a1 = a1 = makeA();  // expected-warning{{variable 'a1' is uninitialized when used within its own initialization}}
  A a2 = a2 == a1;  // expected-warning{{variable 'a2' is uninitialized when used within its own initialization}}
  A a3 = a2 == a3;  // expected-warning{{variable 'a3' is uninitialized when used within its own initialization}}

  int x = x = 5;
}

namespace lambdas {
  struct A {
    template<typename T> A(T) {}
    int x;
  };
  A a0([] { return a0.x; }); // ok
  void f() {
    A a1([=] { // expected-warning{{variable 'a1' is uninitialized when used within its own initialization}}
      return a1.x;
    });
    A a2([&] { return a2.x; }); // ok
  }
}

namespace record_fields {
  bool x;
  struct A {
    A() {}
    A get();
    static A num();
    static A copy(A);
    static A something(A&);
  };

  A ref(A&);
  A const_ref(const A&);
  A pointer(A*);
  A normal(A);
  A rref(A&&);

  struct B {
    A a;
    B(char (*)[1]) : a(a) {}  // expected-warning {{uninitialized}}
    B(char (*)[2]) : a(a.get()) {}  // expected-warning {{uninitialized}}
    B(char (*)[3]) : a(a.num()) {}
    B(char (*)[4]) : a(a.copy(a)) {}  // expected-warning {{uninitialized}}
    B(char (*)[5]) : a(a.something(a)) {}
    B(char (*)[6]) : a(ref(a)) {}
    B(char (*)[7]) : a(const_ref(a)) {}
    B(char (*)[8]) : a(pointer(&a)) {}
    B(char (*)[9]) : a(normal(a)) {}  // expected-warning {{uninitialized}}
    B(char (*)[10]) : a(std::move(a)) {}  // expected-warning {{uninitialized}}
    B(char (*)[11]) : a(A(std::move(a))) {}  // expected-warning {{uninitialized}}
    B(char (*)[12]) : a(rref(std::move(a))) {}  // expected-warning {{uninitialized}}
    B(char (*)[13]) : a(std::move(x ? a : (25, a))) {}  // expected-warning 2{{uninitialized}}
  };
  struct C {
    C() {} // expected-note9{{in this constructor}}
    A a1 = a1;  // expected-warning {{uninitialized}}
    A a2 = a2.get();  // expected-warning {{uninitialized}}
    A a3 = a3.num();
    A a4 = a4.copy(a4);  // expected-warning {{uninitialized}}
    A a5 = a5.something(a5);
    A a6 = ref(a6);
    A a7 = const_ref(a7);
    A a8 = pointer(&a8);
    A a9 = normal(a9);  // expected-warning {{uninitialized}}
    const A a10 = a10;  // expected-warning {{uninitialized}}
    A a11 = std::move(a11);  // expected-warning {{uninitialized}}
    A a12 = A(std::move(a12));  // expected-warning {{uninitialized}}
    A a13 = rref(std::move(a13));  // expected-warning {{uninitialized}}
    A a14 = std::move(x ? a13 : (22, a14));  // expected-warning {{uninitialized}}
  };
  struct D {  // expected-note9{{in the implicit default constructor}}
    A a1 = a1;  // expected-warning {{uninitialized}}
    A a2 = a2.get();  // expected-warning {{uninitialized}}
    A a3 = a3.num();
    A a4 = a4.copy(a4);  // expected-warning {{uninitialized}}
    A a5 = a5.something(a5);
    A a6 = ref(a6);
    A a7 = const_ref(a7);
    A a8 = pointer(&a8);
    A a9 = normal(a9);  // expected-warning {{uninitialized}}
    const A a10 = a10;  // expected-warning {{uninitialized}}
    A a11 = std::move(a11);  // expected-warning {{uninitialized}}
    A a12 = A(std::move(a12));  // expected-warning {{uninitialized}}
    A a13 = rref(std::move(a13));  // expected-warning {{uninitialized}}
    A a14 = std::move(x ? a13 : (22, a14));  // expected-warning {{uninitialized}}
  };
  D d; // expected-note {{in implicit default constructor for 'record_fields::D' first required here}}
  struct E {
    A a1 = a1;
    A a2 = a2.get();
    A a3 = a3.num();
    A a4 = a4.copy(a4);
    A a5 = a5.something(a5);
    A a6 = ref(a6);
    A a7 = const_ref(a7);
    A a8 = pointer(&a8);
    A a9 = normal(a9);
    const A a10 = a10;
    A a11 = std::move(a11);
    A a12 = A(std::move(a12));
    A a13 = rref(std::move(a13));
    A a14 = std::move(x ? a13 : (22, a14));
  };
}

namespace cross_field_warnings {
  struct A {
    int a, b;
    A() {}
    A(char (*)[1]) : b(a) {}  // expected-warning{{field 'a' is uninitialized when used here}}
    A(char (*)[2]) : a(b) {}  // expected-warning{{field 'b' is uninitialized when used here}}
  };

  struct B {
    int a = b;  // expected-warning{{field 'b' is uninitialized when used here}}
    int b;
    B() {} // expected-note{{during field initialization in this constructor}}
  };

  struct C {
    int a;
    int b = a;  // expected-warning{{field 'a' is uninitialized when used here}}
    C(char (*)[1]) : a(5) {}
    C(char (*)[2]) {} // expected-note{{during field initialization in this constructor}}
  };

  struct D {
    int a;
    int &b;
    int &c = a;
    int d = b;
    D() : b(a) {}
  };

  struct E {
    int a;
    int get();
    static int num();
    E() {}
    E(int) {}
  };

  struct F {
    int a;
    E e;
    int b;
    F(char (*)[1]) : a(e.get()) {}  // expected-warning{{field 'e' is uninitialized when used here}}
    F(char (*)[2]) : a(e.num()) {}
    F(char (*)[3]) : e(a) {}  // expected-warning{{field 'a' is uninitialized when used here}}
    F(char (*)[4]) : a(4), e(a) {}
    F(char (*)[5]) : e(b) {}  // expected-warning{{field 'b' is uninitialized when used here}}
    F(char (*)[6]) : e(b), b(4) {}  // expected-warning{{field 'b' is uninitialized when used here}}
  };

  struct G {
    G(const A&) {};
  };

  struct H {
    A a1;
    G g;
    A a2;
    H() : g(a1) {}
    H(int) : g(a2) {}
  };

  struct I {
    I(int*) {}
  };

  struct J : public I {
    int *a;
    int *b;
    int c;
    J() : I((a = new int(5))), b(a), c(*a) {}
  };

  struct K {
    int a = (b = 5);
    int b = b + 5;
  };

  struct L {
    int a = (b = 5);
    int b = b + 5;  // expected-warning{{field 'b' is uninitialized when used here}}
    L() : a(5) {}  // expected-note{{during field initialization in this constructor}}
  };

  struct M { };

  struct N : public M {
    int a;
    int b;
    N() : b(a) { }  // expected-warning{{field 'a' is uninitialized when used here}}
  };

  struct O {
    int x = 42;
    int get() { return x; }
  };

  struct P {
    O o;
    int x = o.get();
    P() : x(o.get()) { }
  };

  struct Q {
    int a;
    int b;
    int &c;
    Q() :
      a(c = 5),  // expected-warning{{reference 'c' is not yet bound to a value when used here}}
      b(c),  // expected-warning{{reference 'c' is not yet bound to a value when used here}}
      c(a) {}
  };

  struct R {
    int a;
    int b;
    int c;
    int d = a + b + c;
    R() : a(c = 5), b(c), c(a) {}
  };

  // FIXME: Use the CFG-based analysis to give a sometimes uninitialized
  // warning on y.
  struct T {
    int x;
    int y;
    T(bool b)
        : x(b ? (y = 5) : (1 + y)),  // expected-warning{{field 'y' is uninitialized when used here}}
          y(y + 1) {}
    T(int b)
        : x(!b ? (1 + y) : (y = 5)),  // expected-warning{{field 'y' is uninitialized when used here}}
          y(y + 1) {}
  };

}

namespace base_class {
  struct A {
    A (int) {}
  };

  struct B : public A {
    int x;
    B() : A(x) {}   // expected-warning{{field 'x' is uninitialized when used here}}
  };

  struct C : public A {
    int x;
    int y;
    C() : A(y = 4), x(y) {}
  };
}

namespace delegating_constructor {
  struct A {
    A(int);
    A(int&, int);

    A(char (*)[1]) : A(x) {}
    // expected-warning@-1 {{field 'x' is uninitialized when used here}}
    A(char (*)[2]) : A(x, x) {}
    // expected-warning@-1 {{field 'x' is uninitialized when used here}}

    A(char (*)[3]) : A(x, 0) {}

    int x;
  };
}

namespace init_list {
  int num = 5;
  struct A { int i1, i2; };
  struct B { A a1, a2; };

  A a1{1,2};
  A a2{a2.i1 + 2};  // expected-warning{{uninitialized}}
  A a3 = {a3.i1 + 2};  // expected-warning{{uninitialized}}
  A a4 = A{a4.i2 + 2};  // expected-warning{{uninitialized}}

  B b1 = { {}, {} };
  B b2 = { {}, b2.a1 };
  B b3 = { b3.a1 };  // expected-warning{{uninitialized}}
  B b4 = { {}, b4.a2} ;  // expected-warning{{uninitialized}}
  B b5 = { b5.a2 };  // expected-warning{{uninitialized}}

  B b6 = { {b6.a1.i1} };  // expected-warning{{uninitialized}}
  B b7 = { {0, b7.a1.i1} };
  B b8 = { {}, {b8.a1.i1} };
  B b9 = { {}, {0, b9.a1.i1} };

  B b10 = { {b10.a1.i2} };  // expected-warning{{uninitialized}}
  B b11 = { {0, b11.a1.i2} };  // expected-warning{{uninitialized}}
  B b12 = { {}, {b12.a1.i2} };
  B b13 = { {}, {0, b13.a1.i2} };

  B b14 = { {b14.a2.i1} };  // expected-warning{{uninitialized}}
  B b15 = { {0, b15.a2.i1} };  // expected-warning{{uninitialized}}
  B b16 = { {}, {b16.a2.i1} };  // expected-warning{{uninitialized}}
  B b17 = { {}, {0, b17.a2.i1} };

  B b18 = { {b18.a2.i2} };  // expected-warning{{uninitialized}}
  B b19 = { {0, b19.a2.i2} };  // expected-warning{{uninitialized}}
  B b20 = { {}, {b20.a2.i2} };  // expected-warning{{uninitialized}}
  B b21 = { {}, {0, b21.a2.i2} };  // expected-warning{{uninitialized}}

  B b22 = { {b18.a2.i2 + 5} };

  struct C {int a; int& b; int c; };
  C c1 = { 0, num, 0 };
  C c2 = { 1, num, c2.b };
  C c3 = { c3.b, num };  // expected-warning{{uninitialized}}
  C c4 = { 0, c4.b, 0 };  // expected-warning{{uninitialized}}
  C c5 = { 0, c5.c, 0 };
  C c6 = { c6.b, num, 0 };  // expected-warning{{uninitialized}}
  C c7 = { 0, c7.a, 0 };

  struct D {int &a; int &b; };
  D d1 = { num, num };
  D d2 = { num, d2.a };
  D d3 = { d3.b, num };  // expected-warning{{uninitialized}}

  // Same as above in member initializer form.
  struct Awrapper {
    A a1{1,2};
    A a2{a2.i1 + 2};  // expected-warning{{uninitialized}}
    A a3 = {a3.i1 + 2};  // expected-warning{{uninitialized}}
    A a4 = A{a4.i2 + 2};  // expected-warning{{uninitialized}}
    Awrapper() {}  // expected-note 3{{in this constructor}}
    Awrapper(int) :
      a1{1,2},
      a2{a2.i1 + 2},  // expected-warning{{uninitialized}}
      a3{a3.i1 + 2},  // expected-warning{{uninitialized}}
      a4{a4.i2 + 2}  // expected-warning{{uninitialized}}
    {}
  };

  struct Bwrapper {
    B b1 = { {}, {} };
    B b2 = { {}, b2.a1 };
    B b3 = { b3.a1 };  // expected-warning{{uninitialized}}
    B b4 = { {}, b4.a2} ;  // expected-warning{{uninitialized}}
    B b5 = { b5.a2 };  // expected-warning{{uninitialized}}

    B b6 = { {b6.a1.i1} };  // expected-warning{{uninitialized}}
    B b7 = { {0, b7.a1.i1} };
    B b8 = { {}, {b8.a1.i1} };
    B b9 = { {}, {0, b9.a1.i1} };

    B b10 = { {b10.a1.i2} };  // expected-warning{{uninitialized}}
    B b11 = { {0, b11.a1.i2} };  // expected-warning{{uninitialized}}
    B b12 = { {}, {b12.a1.i2} };
    B b13 = { {}, {0, b13.a1.i2} };

    B b14 = { {b14.a2.i1} };  // expected-warning{{uninitialized}}
    B b15 = { {0, b15.a2.i1} };  // expected-warning{{uninitialized}}
    B b16 = { {}, {b16.a2.i1} };  // expected-warning{{uninitialized}}
    B b17 = { {}, {0, b17.a2.i1} };

    B b18 = { {b18.a2.i2} };  // expected-warning{{uninitialized}}
    B b19 = { {0, b19.a2.i2} };  // expected-warning{{uninitialized}}
    B b20 = { {}, {b20.a2.i2} };  // expected-warning{{uninitialized}}
    B b21 = { {}, {0, b21.a2.i2} };  // expected-warning{{uninitialized}}

    B b22 = { {b18.a2.i2 + 5} };
    Bwrapper() {}  // expected-note 13{{in this constructor}}
    Bwrapper(int) :
      b1{ {}, {} },
      b2{ {}, b2.a1 },
      b3{ b3.a1 },  // expected-warning{{uninitialized}}
      b4{ {}, b4.a2}, // expected-warning{{uninitialized}}
      b5{ b5.a2 },  // expected-warning{{uninitialized}}

      b6{ {b6.a1.i1} },  // expected-warning{{uninitialized}}
      b7{ {0, b7.a1.i1} },
      b8{ {}, {b8.a1.i1} },
      b9{ {}, {0, b9.a1.i1} },

      b10{ {b10.a1.i2} },  // expected-warning{{uninitialized}}
      b11{ {0, b11.a1.i2} },  // expected-warning{{uninitialized}}
      b12{ {}, {b12.a1.i2} },
      b13{ {}, {0, b13.a1.i2} },

      b14{ {b14.a2.i1} },  // expected-warning{{uninitialized}}
      b15{ {0, b15.a2.i1} },  // expected-warning{{uninitialized}}
      b16{ {}, {b16.a2.i1} },  // expected-warning{{uninitialized}}
      b17{ {}, {0, b17.a2.i1} },

      b18{ {b18.a2.i2} },  // expected-warning{{uninitialized}}
      b19{ {0, b19.a2.i2} },  // expected-warning{{uninitialized}}
      b20{ {}, {b20.a2.i2} },  // expected-warning{{uninitialized}}
      b21{ {}, {0, b21.a2.i2} },  // expected-warning{{uninitialized}}

      b22{ {b18.a2.i2 + 5} }
    {}
  };

  struct Cwrapper {
    C c1 = { 0, num, 0 };
    C c2 = { 1, num, c2.b };
    C c3 = { c3.b, num };  // expected-warning{{uninitialized}}
    C c4 = { 0, c4.b, 0 };  // expected-warning{{uninitialized}}
    C c5 = { 0, c5.c, 0 };
    C c6 = { c6.b, num, 0 };  // expected-warning{{uninitialized}}
    C c7 = { 0, c7.a, 0 };

    Cwrapper() {} // expected-note 3{{in this constructor}}
    Cwrapper(int) :
      c1{ 0, num, 0 },
      c2{ 1, num, c2.b },
      c3{ c3.b, num },  // expected-warning{{uninitialized}}
      c4{ 0, c4.b, 0 },  // expected-warning{{uninitialized}}
      c5{ 0, c5.c, 0 },
      c6{ c6.b, num, 0 },  // expected-warning{{uninitialized}}
      c7{ 0, c7.a, 0 }
    {}
  };

  struct Dwrapper {
    D d1 = { num, num };
    D d2 = { num, d2.a };
    D d3 = { d3.b, num }; // expected-warning{{uninitialized}}
    Dwrapper() {}  // expected-note{{in this constructor}}
    Dwrapper(int) :
      d1{ num, num },
      d2{ num, d2.a },
      d3{ d3.b, num } // expected-warning{{uninitialized}}
    {}
  };
  
  struct E {
    E();
    E foo();
    E* operator->();
  };

  struct F { F(E); };

  struct EFComposed {
    F f;
    E e;
    EFComposed() : f{ e->foo() }, e() {} // expected-warning{{uninitialized}}
  };
}

namespace template_class {
class Foo {
 public:
    int *Create() { return nullptr; }
};

template <typename T>
class A {
public:
  // Don't warn on foo here.
  A() : ptr(foo->Create()) {}

private:
  Foo *foo = new Foo;
  int *ptr;
};

template <typename T>
class B {
public:
  // foo is uninitialized here, but class B is never instantiated.
  B() : ptr(foo->Create()) {}

private:
  Foo *foo;
  int *ptr;
};

template <typename T>
class C {
public:
  C() : ptr(foo->Create()) {}
  // expected-warning@-1 {{field 'foo' is uninitialized when used here}}
private:
  Foo *foo;
  int *ptr;
};

C<int> c;
// expected-note@-1 {{in instantiation of member function 'template_class::C<int>::C' requested here}}

}

namespace base_class_access {
struct A {
  A();
  A(int);

  int i;
  int foo();

  static int bar();
};

struct B : public A {
  B(int (*)[1]) : A() {}
  B(int (*)[2]) : A(bar()) {}

  B(int (*)[3]) : A(i) {}
  // expected-warning@-1 {{base class 'base_class_access::A' is uninitialized when used here to access 'base_class_access::A::i'}}

  B(int (*)[4]) : A(foo()) {}
  // expected-warning@-1 {{base_class_access::A' is uninitialized when used here to access 'base_class_access::A::foo'}}
};

struct C {
  C(int) {}
};

struct D : public C, public A {
  D(int (*)[1]) : C(0) {}
  D(int (*)[2]) : C(bar()) {}

  D(int (*)[3]) : C(i) {}
  // expected-warning@-1 {{base class 'base_class_access::A' is uninitialized when used here to access 'base_class_access::A::i'}}

  D(int (*)[4]) : C(foo()) {}
  // expected-warning@-1 {{base_class_access::A' is uninitialized when used here to access 'base_class_access::A::foo'}}
};

}

namespace value {
template <class T> T move(T t);
template <class T> T notmove(T t);
}
namespace lvalueref {
template <class T> T move(T& t);
template <class T> T notmove(T& t);
}
namespace rvalueref {
template <class T> T move(T&& t);
template <class T> T notmove(T&& t);
}

namespace move_test {
int a1 = std::move(a1); // expected-warning {{uninitialized}}
int a2 = value::move(a2); // expected-warning {{uninitialized}}
int a3 = value::notmove(a3); // expected-warning {{uninitialized}}
int a4 = lvalueref::move(a4);
int a5 = lvalueref::notmove(a5);
int a6 = rvalueref::move(a6);
int a7 = rvalueref::notmove(a7);

void test() {
  int a1 = std::move(a1); // expected-warning {{uninitialized}}
  int a2 = value::move(a2); // expected-warning {{uninitialized}}
  int a3 = value::notmove(a3); // expected-warning {{uninitialized}}
  int a4 = lvalueref::move(a4);
  int a5 = lvalueref::notmove(a5);
  int a6 = rvalueref::move(a6);
  int a7 = rvalueref::notmove(a7);
}

class A {
  int a;
  A(int (*) [1]) : a(std::move(a)) {} // expected-warning {{uninitialized}}
  A(int (*) [2]) : a(value::move(a)) {} // expected-warning {{uninitialized}}
  A(int (*) [3]) : a(value::notmove(a)) {} // expected-warning {{uninitialized}}
  A(int (*) [4]) : a(lvalueref::move(a)) {}
  A(int (*) [5]) : a(lvalueref::notmove(a)) {}
  A(int (*) [6]) : a(rvalueref::move(a)) {}
  A(int (*) [7]) : a(rvalueref::notmove(a)) {}
};
}

void array_capture(bool b) {
  const char fname[] = "array_capture";
  if (b) {
    int unused; // expected-warning {{unused variable}}
  } else {
    [fname]{};
  }
}

void if_switch_init_stmt(int k) {
  if (int n = 0; (n == k || k > 5)) {}

  if (int n; (n == k || k > 5)) {} // expected-warning {{uninitialized}} expected-note {{initialize}}

  switch (int n = 0; (n == k || k > 5)) {} // expected-warning {{boolean}}

  switch (int n; (n == k || k > 5)) {} // expected-warning {{uninitialized}} expected-note {{initialize}} expected-warning {{boolean}}
}

template<typename T> struct Outer {
  struct Inner {
    int a = 1;
    int b;
    Inner() : b(a) {}
  };
};
Outer<int>::Inner outerinner;

struct Polymorphic { virtual ~Polymorphic() { } };

template<class... Bases>
struct Inherit : Bases... { // #TYPE_INHERIT
  int g1; // #FIELD_G1
};

template<class... Bases>
struct InheritWithExplicit : Bases... { // #TYPE_INHERIT_WITH_EXPLICIT
  int g2 [[clang::require_explicit_initialization]]; // #FIELD_G2
};

struct Special {};

template<>
struct Inherit<Special> {
  int g3 [[clang::require_explicit_initialization]]; // #FIELD_G3
};

template<>
struct InheritWithExplicit<Special> {
  int g4; // #FIELD_G4
};

void aggregate() {
  struct NonAgg {
    NonAgg() { }
    [[clang::require_explicit_initialization]] int na;  // expected-warning {{'require_explicit_initialization' attribute is ignored in non-aggregate type 'NonAgg'}}
  };
  NonAgg nonagg;  // no-warning
  (void)nonagg;

  struct S {
    [[clang::require_explicit_initialization]] int s1; // #FIELD_S1
    int s2;
    int s3 = 12;
    [[clang::require_explicit_initialization]] int s4 = 100; // #FIELD_S4
    static void foo(S) { }
  };

  struct C {
#if __cplusplus < 202002L
    // expected-warning@+1 {{explicit initialization of field 'c1' will not be enforced in C++20 and later because 'C' has a user-declared constructor, making the type no longer an aggregate}}
    [[clang::require_explicit_initialization]]
#endif
    int c1; // #FIELD_C1
    C() = default;  // Test pre-C++20 aggregates
  };

  struct D : S { // #TYPE_D
    int d1;
    int d2 [[clang::require_explicit_initialization]]; // #FIELD_D2
  };

  struct D2 : D {  // #TYPE_D2
  };

  struct E {  // #TYPE_E
    int e1;
    D e2 [[clang::require_explicit_initialization]]; // #FIELD_E2
    struct {
      [[clang::require_explicit_initialization]] D e3;
      D2 e4 [[clang::require_explicit_initialization]];
    };
  };

  struct CopyAndMove {
    CopyAndMove() = default;
    CopyAndMove(const CopyAndMove &) {}
    CopyAndMove(CopyAndMove &&) {}
  };
  struct Embed {
    int embed1;  // #FIELD_EMBED1
    int embed2 [[clang::require_explicit_initialization]];  // #FIELD_EMBED2
    CopyAndMove force_separate_move_ctor;
  };
  struct EmbedDerived : Embed {};
  struct F {
    Embed f1;
    // expected-warning@+1 {{field in 'Embed' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
    explicit F(const char(&)[1]) : f1() {
      // expected-warning@+1 {{field in 'Embed' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
      ::new(static_cast<void*>(&f1)) decltype(f1);
      // expected-warning@+1 {{field in 'Embed' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
      ::new(static_cast<void*>(&f1)) decltype(f1)();
#if __cplusplus >= 202002L
      // expected-warning@+1 {{field 'embed2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
      ::new(static_cast<void*>(&f1)) decltype(f1)(1);
#endif
      // expected-warning@+1 {{field 'embed2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
      ::new(static_cast<void*>(&f1)) decltype(f1){1};
    }
#if __cplusplus >= 202002L
    // expected-warning@+1 {{field 'embed2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
    explicit F(const char(&)[2]) : f1(1) {}
#else
    explicit F(const char(&)[2]) : f1{1, 2} { }
#endif
    // expected-warning@+1 {{field 'embed2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
    explicit F(const char(&)[3]) : f1{} {}
    // expected-warning@+1 {{field 'embed2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
    explicit F(const char(&)[4]) : f1{1} {}
    // expected-warning@+1 {{field 'embed2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_EMBED2 {{'embed2' declared here}}
    explicit F(const char(&)[5]) : f1{.embed1 = 1} {}
  };
  F ctors[] = {
      F(""),
      F("_"),
      F("__"),
      F("___"),
      F("____")
  };

  struct MoveOrCopy {
    Embed e;
    EmbedDerived ed;
    F f;
    // no-error
    MoveOrCopy(const MoveOrCopy &c) : e(c.e), ed(c.ed), f(c.f) {}
    // no-error
    MoveOrCopy(MoveOrCopy &&c)
        : e(std::move(c.e)), ed(std::move(c.ed)), f(std::move(c.f)) {}
  };
  F copy1(ctors[0]); // no-error
  (void)copy1;
  F move1(std::move(ctors[0])); // no-error
  (void)move1;
  F copy2{ctors[0]}; // no-error
  (void)copy2;
  F move2{std::move(ctors[0])}; // no-error
  (void)move2;
  F copy3 = ctors[0]; // no-error
  (void)copy3;
  F move3 = std::move(ctors[0]); // no-error
  (void)move3;
  F copy4 = {ctors[0]}; // no-error
  (void)copy4;
  F move4 = {std::move(ctors[0])}; // no-error
  (void)move4;

  S::foo(S{1, 2, 3, 4});
  S::foo(S{.s1 = 100, .s4 = 100});
  S::foo(S{.s1 = 100}); // expected-warning {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}

  S s{.s1 = 100, .s4 = 100};
  (void)s;

  S t{.s4 = 100}; // expected-warning {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  (void)t;

  S *ptr1 = new S; // expected-warning {{field in 'S' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}} expected-note@#FIELD_S1 {{'s1' declared here}}
  delete ptr1;

  S *ptr2 = new S{.s1 = 100, .s4 = 100};
  delete ptr2;

#if __cplusplus >= 202002L
  // expected-warning@+3 {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D a1({}, 0);
  (void)a1;

  // expected-warning@+3 {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D a2(S{}, 0);
  (void)a2;

  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D a3(S{.s1 = 0}, 0);
  (void)a3;

  // expected-warning@+3 {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D a4(S(), 0);
  (void)a4;

  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D a5(S(0), 0);
  (void)a5;

  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D a6 = {S(0), 0};
  (void)a6;
#endif

#if 201103L <= __cplusplus && __cplusplus < 202002L
  C a; // expected-warning {{field in 'C' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_C1 {{'c1' declared here}}
  (void)a;
#endif

  // expected-warning@+2 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  D b{.d2 = 1};
  (void)b;

  // expected-warning@+3 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+2 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  // expected-warning@+1 {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  D c{.d1 = 5};

  // expected-warning@+3 {{field 's4' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+2 {{field 'd2' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  // expected-warning@+1 {{field 's1' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}}
  c = {{}, 0};
  (void)c;

  // expected-note@+3 {{in implicit default constructor for 'D' first required here}}
  // expected-warning@#TYPE_D {{field in 'S' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_S1 {{'s1' declared here}} expected-note@#FIELD_S4 {{'s4' declared here}}
  // expected-warning@+1 {{field in 'D' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_D2 {{'d2' declared here}}
  D d;
  (void)d;

  // expected-warning@+12 {{field in 'E' requires explicit initialization but is not explicitly initialized}}
  // expected-note@#FIELD_E2 {{'e2' declared here}}
  // expected-warning@#TYPE_E {{field in 'D' requires explicit initialization but is not explicitly initialized}}
  // expected-note@+9 {{in implicit default constructor for 'E' first required here}}
  // expected-note@#FIELD_D2 {{'d2' declared here}}
  // expected-warning@#TYPE_E {{field in 'D' requires explicit initialization but is not explicitly initialized}}
  // expected-note@#FIELD_D2 {{'d2' declared here}}
  // expected-warning@#TYPE_E {{field in 'D2' requires explicit initialization but is not explicitly initialized}}
  // expected-note@#TYPE_E {{in implicit default constructor for 'D2' first required here}}
  // expected-warning@#TYPE_D2 {{field in 'D' requires explicit initialization but is not explicitly initialized}}
  // expected-note@+2 {{in implicit default constructor for 'E' first required here}}
  // expected-note@#FIELD_D2 {{'d2' declared here}}
  E e;
  (void)e;

  InheritWithExplicit<> agg;  // expected-warning {{field in 'InheritWithExplicit<>' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_G2 {{'g2' declared here}}
  (void)agg;

  InheritWithExplicit<Polymorphic> polymorphic;  // expected-warning@#FIELD_G2 {{'require_explicit_initialization' attribute is ignored in non-aggregate type 'InheritWithExplicit<Polymorphic>'}}
  (void)polymorphic;

  Inherit<Special> specialized_explicit;  // expected-warning {{field in 'Inherit<Special>' requires explicit initialization but is not explicitly initialized}} expected-note@#FIELD_G3 {{'g3' declared here}}
  (void)specialized_explicit;

  InheritWithExplicit<Special> specialized_implicit;  // no-warning
  (void)specialized_implicit;
}
