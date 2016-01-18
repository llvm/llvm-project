// RUN: %clang_cc1 -fsyntax-only -fshow-overloads=best -verify -triple x86_64-linux-gnu %s

struct yes;
struct no;

struct Short {
  operator short();
};

struct Long {
  operator long();
};

enum E1 { };
struct Enum1 {
  operator E1();
};

enum E2 { };
struct Enum2 {
  operator E2();
};


struct X { 
  void f();
};

typedef void (X::*pmf)();
struct Xpmf {
  operator pmf();
};

yes& islong(long);
yes& islong(unsigned long); // FIXME: shouldn't be needed
no& islong(int);

void f(Short s, Long l, Enum1 e1, Enum2 e2, Xpmf pmf) {
  // C++ [over.built]p8
  int i1 = +e1;
  int i2 = -e2;

  // C++  [over.built]p10:
  int i3 = ~s;
  bool b1 = !s;

  // C++ [over.built]p12
  (void)static_cast<yes&>(islong(s + l));
  (void)static_cast<no&>(islong(s + s));

  // C++ [over.built]p16
  (void)(pmf == &X::f);
  (void)(pmf == 0);
  
  // C++ [over.built]p17
  (void)static_cast<yes&>(islong(s % l));
  (void)static_cast<yes&>(islong(l << s));
  (void)static_cast<no&>(islong(s << l));
  (void)static_cast<yes&>(islong(e1 % l));
  // FIXME: should pass (void)static_cast<no&>(islong(e1 % e2));
}

struct ShortRef { // expected-note{{candidate function (the implicit copy assignment operator)}}
  operator short&();
};

struct LongRef {
  operator volatile long&();
};

struct XpmfRef { // expected-note{{candidate function (the implicit copy assignment operator)}}
  operator pmf&();
};

struct E2Ref {
  operator E2&();
};

void g(ShortRef sr, LongRef lr, E2Ref e2_ref, XpmfRef pmf_ref) {
  // C++ [over.built]p3
  short s1 = sr++;

  // C++ [over.built]p3
  long l1 = lr--;

  // C++ [over.built]p18
  short& sr1 = (sr *= lr);
  volatile long& lr1 = (lr *= sr);

  // C++ [over.built]p20:
  E2 e2r2;
  e2r2 = e2_ref;
  
  pmf &pmr = (pmf_ref = &X::f); // expected-error{{no viable overloaded '='}}
  pmf pmr2;
  pmr2 = pmf_ref;
               
  // C++ [over.built]p22
  short& sr2 = (sr %= lr);
  volatile long& lr2 = (lr <<= sr);

  bool b1 = (sr && lr) || (sr || lr);
}

struct VolatileIntPtr {
  operator int volatile *();
};

struct ConstIntPtr {
  operator int const *();
};

struct VolatileIntPtrRef {
  operator int volatile *&();
};

struct ConstIntPtrRef {
  operator int const *&();
};

void test_with_ptrs(VolatileIntPtr vip, ConstIntPtr cip, ShortRef sr,
                    VolatileIntPtrRef vipr, ConstIntPtrRef cipr) {
  const int& cir1 = cip[sr];
  const int& cir2 = sr[cip];
  volatile int& vir1 = vip[sr];
  volatile int& vir2 = sr[vip];
  bool b1 = (vip == cip);
  long p1 = vip - cip;

  // C++ [over.built]p5:
  int volatile *vip1 = vipr++;
  int const *cip1 = cipr++;
  int volatile *&vipr1 = ++vipr;
  int const *&cipr1 = --cipr;

  // C++ [over.built]p6:
  int volatile &ivr = *vip;

  // C++ [over.built]p8:
  int volatile *vip2 = +vip;
  int i1 = +sr;
  int i2 = -sr;

  // C++ [over.built]p13:
  int volatile &ivr2 = vip[17];
  int const &icr2 = 17[cip];
}

// C++ [over.match.open]p4

void test_assign_restrictions(ShortRef& sr) {
  sr = (short)0; // expected-error{{no viable overloaded '='}}
}

struct Base { };
struct Derived1 : Base { };
struct Derived2 : Base { };

template<typename T>
struct ConvertibleToPtrOf {
  operator T*();
};

bool test_with_base_ptrs(ConvertibleToPtrOf<Derived1> d1, 
                         ConvertibleToPtrOf<Derived2> d2) {
  return d1 == d2; // expected-error{{invalid operands}}
}

// DR425
struct A {
  template< typename T > operator T() const;
};

void test_dr425(A a) {
  // FIXME: lots of candidates here!
  (void)(1.0f * a); // expected-error{{ambiguous}} \
                    // expected-note 4{{candidate}} \
                    // expected-note {{remaining 117 candidates omitted; pass -fshow-overloads=all to show them}}
}

// pr5432
enum e {X};

const int a[][2] = {{1}};

int test_pr5432() {
  return a[X][X];
}

void f() {
  (void)__extension__(A());
}

namespace PR7319 {
  typedef enum { Enum1, Enum2, Enum3 } MyEnum;

  template<typename X> bool operator>(const X &inX1, const X &inX2);

  void f() {
    MyEnum e1, e2;
    if (e1 > e2) {}
  }
}

namespace PR8477 {
  struct Foo {
    operator bool();
    operator const char *();
  };

  bool doit() {
    Foo foo;
    long long zero = 0;
    (void)(foo + zero);
    (void)(foo - zero);
    (void)(zero + foo);
    (void)(zero[foo]);
    (void)(foo - foo); // expected-error{{use of overloaded operator '-' is ambiguous}} \
    // expected-note 4{{built-in candidate operator-}} \
    // expected-note{{candidates omitted}}
    return foo[zero] == zero;
  }
}

namespace PR7851 {
  struct X {
    operator const void *() const;
    operator void *();

    operator const unsigned *() const;
    operator unsigned *();
  };

  void f() {
    X x;
    x[0] = 1;
    *x = 0;
    (void)(x - x);
  }
}

namespace PR12854 {
  enum { size = 1 };
  void plus_equals() {
    int* __restrict py;
    py += size;
  }

  struct RestrictInt {
    operator int* __restrict &();
  };

  void user_conversions(RestrictInt ri) {
    ++ri;
    --ri;
    ri++;
    ri--;
  }
}

namespace PR12964 {
  struct X { operator  __int128() const; } x;
  bool a = x == __int128(0);
  bool b = x == 0;

  struct Y { operator unsigned __int128() const; } y;
  bool c = y == __int128(0);
  bool d = y == 0;

  bool e = x == y;
}
