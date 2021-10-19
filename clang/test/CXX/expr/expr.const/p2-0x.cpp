// RUN: %clang_cc1 -fsyntax-only -std=c++11 -pedantic -verify=expected,cxx11 -fcxx-exceptions %s -fconstexpr-depth 128 -triple i686-pc-linux-gnu
// RUN: %clang_cc1 -fsyntax-only -std=c++2a -pedantic -verify=expected,cxx20 -fcxx-exceptions %s -fconstexpr-depth 128 -triple i686-pc-linux-gnu

// A conditional-expression is a core constant expression unless it involves one
// of the following as a potentially evaluated subexpression [...]:

// - this (5.1.1 [expr.prim.general]) [Note: when evaluating a constant
//   expression, function invocation substitution (7.1.5 [dcl.constexpr])
//   replaces each occurrence of this in a constexpr member function with a
//   pointer to the class object. -end note];
struct This {
  int this1 : this1; // expected-error {{undeclared}}
  int this2 : this->this1; // expected-error {{invalid}}
  void this3() {
    int n1[this->this1]; // expected-warning {{variable length array}} expected-note {{'this'}}
    int n2[this1]; // expected-warning {{variable length array}} expected-note {{'this'}}
    (void)n1, (void)n2;
  }
};

// - an invocation of a function other than a constexpr constructor for a
//   literal class or a constexpr function [ Note: Overload resolution (13.3)
//   is applied as usual - end note ];
struct NonConstexpr1 {
  static int f() { return 1; } // expected-note {{here}}
  int n : f(); // expected-error {{constant expression}} expected-note {{non-constexpr function 'f' cannot be used in a constant expression}}
};
struct NonConstexpr2 {
  constexpr NonConstexpr2(); // expected-note {{here}}
  int n;
};
struct NonConstexpr3 {
  NonConstexpr3();
  int m : NonConstexpr2().n; // expected-error {{constant expression}} expected-note {{undefined constructor 'NonConstexpr2'}}
};
struct NonConstexpr4 {
  NonConstexpr4();
  int n;
};
struct NonConstexpr5 {
  int n : NonConstexpr4().n; // expected-error {{constant expression}} expected-note {{non-literal type 'NonConstexpr4' cannot be used in a constant expression}}
};

// - an invocation of an undefined constexpr function or an undefined
//   constexpr constructor;
struct UndefinedConstexpr {
  constexpr UndefinedConstexpr();
  static constexpr int undefinedConstexpr1(); // expected-note {{here}}
  int undefinedConstexpr2 : undefinedConstexpr1(); // expected-error {{constant expression}} expected-note {{undefined function 'undefinedConstexpr1' cannot be used in a constant expression}}
};

// - an invocation of a constexpr function with arguments that, when substituted
//   by function invocation substitution (7.1.5), do not produce a core constant
//   expression;
namespace NonConstExprReturn {
  static constexpr const int &id_ref(const int &n) {
    return n;
  }
  struct NonConstExprFunction {
    int n : id_ref(16); // ok
  };
  constexpr const int *address_of(const int &a) {
    return &a;
  }
  constexpr const int *return_param(int n) {
    return address_of(n);
  }
  struct S {
    int n : *return_param(0); // expected-error {{constant expression}} expected-note {{read of object outside its lifetime}}
  };
}

// - an invocation of a constexpr constructor with arguments that, when
//   substituted by function invocation substitution (7.1.5), do not produce all
//   constant expressions for the constructor calls and full-expressions in the
//   mem-initializers (including conversions);
namespace NonConstExprCtor {
  struct T {
    constexpr T(const int &r) :
      r(r) {
    }
    const int &r;
  };
  constexpr int n = 0;
  constexpr T t1(n); // ok
  constexpr T t2(0); // expected-error {{must be initialized by a constant expression}} expected-note {{temporary created here}} expected-note {{reference to temporary is not a constant expression}}

  struct S {
    int n : T(4).r; // ok
  };
}

// - an invocation of a constexpr function or a constexpr constructor that would
//   exceed the implementation-defined recursion limits (see Annex B);
namespace RecursionLimits {
  constexpr int RecurseForever(int n) {
    return n + RecurseForever(n+1); // expected-note {{constexpr evaluation exceeded maximum depth of 128 calls}} expected-note 9{{in call to 'RecurseForever(}} expected-note {{skipping 118 calls}}
  }
  struct AlsoRecurseForever {
    constexpr AlsoRecurseForever(int n) :
      n(AlsoRecurseForever(n+1).n) // expected-note {{constexpr evaluation exceeded maximum depth of 128 calls}} expected-note 9{{in call to 'AlsoRecurseForever(}} expected-note {{skipping 118 calls}}
    {}
    int n;
  };
  struct S {
    int k : RecurseForever(0); // expected-error {{constant expression}} expected-note {{in call to}}
    int l : AlsoRecurseForever(0).n; // expected-error {{constant expression}} expected-note {{in call to}}
  };
}

// DR1458: taking the address of an object of incomplete class type
namespace IncompleteClassTypeAddr {
  struct S;
  extern S s;
  constexpr S *p = &s; // ok
  static_assert(p, "");

  extern S sArr[];
  constexpr S (*p2)[] = &sArr; // ok

  struct S {
    constexpr S *operator&() const { return nullptr; }
  };
  constexpr S *q = &s; // ok
  static_assert(!q, "");
}

// - an operation that would have undefined behavior [Note: including, for
//   example, signed integer overflow (Clause 5 [expr]), certain pointer
//   arithmetic (5.7 [expr.add]), division by zero (5.6 [expr.mul]), or certain
//   shift operations (5.8 [expr.shift]) -end note];
namespace UndefinedBehavior {
  void f(int n) {
    switch (n) {
    case (int)4.4e9: // expected-error {{constant expression}} expected-note {{value 4.4E+9 is outside the range of representable values of type 'int'}}
    case (int)0x80000000u: // ok
    case (int)10000000000ll: // expected-note {{here}}
    case (unsigned int)10000000000ll: // expected-error {{duplicate case value}}
    case (int)(unsigned)(long long)4.4e9: // ok
    case (int)(float)1e300: // expected-error {{constant expression}} expected-note {{value +Inf is outside the range of representable values of type 'int'}}
    case (int)((float)1e37 / 1e30): // ok
    case (int)(__fp16)65536: // expected-error {{constant expression}} expected-note {{value +Inf is outside the range of representable values of type 'int'}}
      break;
    }
  }

  constexpr int int_min = ~0x7fffffff;
  constexpr int minus_int_min = -int_min; // expected-error {{constant expression}} expected-note {{value 2147483648 is outside the range}}
  constexpr int div0 = 3 / 0; // expected-error {{constant expression}} expected-note {{division by zero}}
  constexpr int mod0 = 3 % 0; // expected-error {{constant expression}} expected-note {{division by zero}}
  constexpr int int_min_div_minus_1 = int_min / -1; // expected-error {{constant expression}} expected-note {{value 2147483648 is outside the range}}
  constexpr int int_min_mod_minus_1 = int_min % -1; // expected-error {{constant expression}} expected-note {{value 2147483648 is outside the range}}

  constexpr int shl_m1 = 0 << -1; // expected-error {{constant expression}} expected-note {{negative shift count -1}}
  constexpr int shl_0 = 0 << 0; // ok
  constexpr int shl_31 = 0 << 31; // ok
  constexpr int shl_32 = 0 << 32; // expected-error {{constant expression}} expected-note {{shift count 32 >= width of type 'int' (32}}
  constexpr int shl_unsigned_negative = unsigned(-3) << 1; // ok
  constexpr int shl_unsigned_into_sign = 1u << 31; // ok
  constexpr int shl_unsigned_overflow = 1024u << 31; // ok
  constexpr int shl_signed_negative = (-3) << 1; // cxx11-error {{constant expression}} cxx11-note {{left shift of negative value -3}}
  constexpr int shl_signed_ok = 1 << 30; // ok
  constexpr int shl_signed_into_sign = 1 << 31; // ok (DR1457)
  constexpr int shl_signed_into_sign_2 = 0x7fffffff << 1; // ok (DR1457)
  constexpr int shl_signed_off_end = 2 << 31; // cxx11-error {{constant expression}} cxx11-note {{signed left shift discards bits}} expected-warning {{signed shift result (0x100000000) requires 34 bits to represent, but 'int' only has 32 bits}}
  constexpr int shl_signed_off_end_2 = 0x7fffffff << 2; // cxx11-error {{constant expression}} cxx11-note {{signed left shift discards bits}} expected-warning {{signed shift result (0x1FFFFFFFC) requires 34 bits to represent, but 'int' only has 32 bits}}
  constexpr int shl_signed_overflow = 1024 << 31; // cxx11-error {{constant expression}} cxx11-note {{signed left shift discards bits}} expected-warning {{requires 43 bits to represent}}
  constexpr int shl_signed_ok2 = 1024 << 20; // ok

  constexpr int shr_m1 = 0 >> -1; // expected-error {{constant expression}} expected-note {{negative shift count -1}}
  constexpr int shr_0 = 0 >> 0; // ok
  constexpr int shr_31 = 0 >> 31; // ok
  constexpr int shr_32 = 0 >> 32; // expected-error {{constant expression}} expected-note {{shift count 32 >= width of type}}

  struct S {
    int m;
  };
  constexpr S s = { 5 };
  constexpr const int *p = &s.m + 1;
  constexpr const int &f(const int *q) {
    return q[0];
  }
  constexpr int n = (f(p), 0); // ok
  struct T {
    int n : f(p); // expected-error {{not an integral constant expression}} expected-note {{read of dereferenced one-past-the-end pointer}}
  };

  namespace Ptr {
    struct A {};
    struct B : A { int n; };
    B a[3][3];
    constexpr B *p = a[0] + 4; // expected-error {{constant expression}} expected-note {{element 4 of array of 3 elements}}
    B b = {};
    constexpr A *pa = &b + 1; // expected-error {{constant expression}} expected-note {{base class of pointer past the end}}
    constexpr B *pb = (B*)((A*)&b + 1); // expected-error {{constant expression}} expected-note {{derived class of pointer past the end}}
    constexpr const int *pn = &(&b + 1)->n; // expected-error {{constant expression}} expected-note {{field of pointer past the end}}
    constexpr B *parr = &a[3][0]; // expected-error {{constant expression}} expected-note {{array element of pointer past the end}}

    constexpr A *na = nullptr;
    constexpr B *nb = nullptr;
    constexpr A &ra = *nb; // expected-error {{constant expression}} expected-note {{cannot access base class of null pointer}}
    constexpr B &rb = (B&)*na; // expected-error {{constant expression}} expected-note {{cannot access derived class of null pointer}}
    static_assert((A*)nb == 0, "");
    static_assert((B*)na == 0, "");
    constexpr const int &nf = nb->n; // expected-error {{constant expression}} expected-note {{cannot access field of null pointer}}
    constexpr const int *np1 = (int*)nullptr + 0; // ok
    constexpr const int *np2 = &(*(int(*)[4])nullptr)[0]; // ok
    constexpr const int *np3 = &(*(int(*)[4])nullptr)[2]; // expected-error {{constant expression}} expected-note {{cannot perform pointer arithmetic on null pointer}}

    struct C {
      constexpr int f() const { return 0; }
    } constexpr c = C();
    constexpr int k1 = c.f(); // ok
    constexpr int k2 = ((C*)nullptr)->f(); // expected-error {{constant expression}} expected-note {{member call on dereferenced null pointer}}
    constexpr int k3 = (&c)[1].f(); // expected-error {{constant expression}} expected-note {{member call on dereferenced one-past-the-end pointer}}
    C c2;
    constexpr int k4 = c2.f(); // ok!

    constexpr int diff1 = &a[2] - &a[0];
    constexpr int diff2 = &a[1][3] - &a[1][0];
    constexpr int diff3 = &a[2][0] - &a[1][0]; // expected-error {{constant expression}} expected-note {{subtracted pointers are not elements of the same array}}
    static_assert(&a[2][0] == &a[1][3], "");
    constexpr int diff4 = (&b + 1) - &b;
    constexpr int diff5 = &a[1][2].n - &a[1][0].n; // expected-error {{constant expression}} expected-note {{subtracted pointers are not elements of the same array}}
    constexpr int diff6 = &a[1][2].n - &a[1][2].n;
    constexpr int diff7 = (A*)&a[0][1] - (A*)&a[0][0]; // expected-error {{constant expression}} expected-note {{subtracted pointers are not elements of the same array}}
  }

  namespace Overflow {
    // Signed int overflow.
    constexpr int n1 = 2 * 3 * 3 * 7 * 11 * 31 * 151 * 331; // ok
    constexpr int n2 = 65536 * 32768; // expected-error {{constant expression}} expected-note {{value 2147483648 is outside the range of }}
    constexpr int n3 = n1 + 1; // ok
    constexpr int n4 = n3 + 1; // expected-error {{constant expression}} expected-note {{value 2147483648 is outside the range of }}
    constexpr int n5 = -65536 * 32768; // ok
    constexpr int n6 = 3 * -715827883; // expected-error {{constant expression}} expected-note {{value -2147483649 is outside the range of }}
    constexpr int n7 = -n3 + -1; // ok
    constexpr int n8 = -1 + n7; // expected-error {{constant expression}} expected-note {{value -2147483649 is outside the range of }}
    constexpr int n9 = n3 - 0; // ok
    constexpr int n10 = n3 - -1; // expected-error {{constant expression}} expected-note {{value 2147483648 is outside the range of }}
    constexpr int n11 = -1 - n3; // ok
    constexpr int n12 = -2 - n3; // expected-error {{constant expression}} expected-note {{value -2147483649 is outside the range of }}
    constexpr int n13 = n5 + n5; // expected-error {{constant expression}} expected-note {{value -4294967296 is outside the range of }}
    constexpr int n14 = n3 - n5; // expected-error {{constant expression}} expected-note {{value 4294967295 is outside the range of }}
    constexpr int n15 = n5 * n5; // expected-error {{constant expression}} expected-note {{value 4611686018427387904 is outside the range of }}
    constexpr signed char c1 = 100 * 2; // ok expected-warning{{changes value}}
    constexpr signed char c2 = '\x64' * '\2'; // also ok  expected-warning{{changes value}}
    constexpr long long ll1 = 0x7fffffffffffffff; // ok
    constexpr long long ll2 = ll1 + 1; // expected-error {{constant}} expected-note {{ 9223372036854775808 }}
    constexpr long long ll3 = -ll1 - 1; // ok
    constexpr long long ll4 = ll3 - 1; // expected-error {{constant}} expected-note {{ -9223372036854775809 }}
    constexpr long long ll5 = ll3 * ll3; // expected-error {{constant}} expected-note {{ 85070591730234615865843651857942052864 }}

    // Yikes.
    char melchizedek[2200000000];
    typedef decltype(melchizedek[1] - melchizedek[0]) ptrdiff_t;
    constexpr ptrdiff_t d1 = &melchizedek[0x7fffffff] - &melchizedek[0]; // ok
    constexpr ptrdiff_t d2 = &melchizedek[0x80000000u] - &melchizedek[0]; // expected-error {{constant expression}} expected-note {{ 2147483648 }}
    constexpr ptrdiff_t d3 = &melchizedek[0] - &melchizedek[0x80000000u]; // ok
    constexpr ptrdiff_t d4 = &melchizedek[0] - &melchizedek[0x80000001u]; // expected-error {{constant expression}} expected-note {{ -2147483649 }}

    // Unsigned int overflow.
    static_assert(65536u * 65536u == 0u, ""); // ok
    static_assert(4294967295u + 1u == 0u, ""); // ok
    static_assert(0u - 1u == 4294967295u, ""); // ok
    static_assert(~0u * ~0u == 1u, ""); // ok

    template<typename T> constexpr bool isinf(T v) { return v && v / 2 == v; }

    // Floating-point overflow and NaN.
    constexpr float f1 = 1e38f * 3.4028f; // ok
    constexpr float f2 = 1e38f * 3.4029f; // ok, +inf is in range of representable values
    constexpr float f3 = 1e38f / -.2939f; // ok
    constexpr float f4 = 1e38f / -.2938f; // ok, -inf is in range of representable values
    constexpr float f5 = 2e38f + 2e38f; // ok, +inf is in range of representable values
    constexpr float f6 = -2e38f - 2e38f; // ok, -inf is in range of representable values
    constexpr float f7 = 0.f / 0.f; // expected-error {{constant expression}} expected-note {{division by zero}}
    constexpr float f8 = 1.f / 0.f; // expected-error {{constant expression}} expected-note {{division by zero}}
    constexpr float f9 = 1e308 / 1e-308; // ok, +inf
    constexpr float f10 = f2 - f2; // expected-error {{constant expression}} expected-note {{produces a NaN}}
    constexpr float f11 = f2 + f4; // expected-error {{constant expression}} expected-note {{produces a NaN}}
    constexpr float f12 = f2 / f2; // expected-error {{constant expression}} expected-note {{produces a NaN}}
#pragma float_control(push)
#pragma float_control(except, on)
constexpr float pi = 3.14f;
constexpr unsigned ubig = 0xFFFFFFFF;
constexpr float ce = 1.0 / 3.0; // not-expected-error {{constant expression}} not-expected-note {{floating point arithmetic suppressed in strict evaluation modes}}
constexpr int ci = (int) pi;
constexpr float fbig = (float) ubig; // not-expected-error {{constant expression}} not-expected-note {{floating point arithmetic suppressed in strict evaluation modes}}
constexpr float fabspi = __builtin_fabs(pi); // no error expected
constexpr float negpi = -pi; // expect no error on unary operator
#pragma float_control(pop)
    static_assert(!isinf(f1), "");
    static_assert(isinf(f2), "");
    static_assert(!isinf(f3), "");
    static_assert(isinf(f4), "");
    static_assert(isinf(f5), "");
    static_assert(isinf(f6), "");
    static_assert(isinf(f9), "");
  }
}

// - a lambda-expression (5.1.2);
struct Lambda {
  int n : []{ return 1; }(); // cxx11-error {{constant expression}} cxx11-error {{integral constant expression}} cxx11-note {{non-literal type}}
};

// - an lvalue-to-rvalue conversion (4.1) unless it is applied to
namespace LValueToRValue {
  // - a non-volatile glvalue of integral or enumeration type that refers to a
  //   non-volatile const object with a preceding initialization, initialized
  //   with a constant expression  [Note: a string literal (2.14.5 [lex.string])
  //   corresponds to an array of such objects. -end note], or
  volatile const int vi = 1; // expected-note 2{{here}}
  const int ci = 1;
  volatile const int &vrci = ci;
  static_assert(vi, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
  static_assert(const_cast<int&>(vi), ""); // expected-error {{constant expression}} expected-note {{read of volatile object 'vi'}}
  static_assert(vrci, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}

  // - a non-volatile glvalue of literal type that refers to a non-volatile
  //   object defined with constexpr, or that refers to a sub-object of such an
  //   object, or
  struct V {
    constexpr V() : v(1) {}
    volatile int v; // expected-note {{not literal because}}
  };
  constexpr V v; // expected-error {{non-literal type}}
  struct S {
    constexpr S(int=0) : i(1), v(const_cast<volatile int&>(vi)) {}
    constexpr S(const S &s) : i(2), v(const_cast<volatile int&>(vi)) {}
    int i;
    volatile int &v;
  };
  constexpr S s; // ok
  constexpr volatile S vs; // expected-note {{here}}
  constexpr const volatile S &vrs = s; // ok
  static_assert(s.i, "");
  static_assert(s.v, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
  static_assert(const_cast<int&>(s.v), ""); // expected-error {{constant expression}} expected-note {{read of volatile object 'vi'}}
  static_assert(vs.i, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
  static_assert(const_cast<int&>(vs.i), ""); // expected-error {{constant expression}} expected-note {{read of volatile object 'vs'}}
  static_assert(vrs.i, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}

  // - a non-volatile glvalue of literal type that refers to a non-volatile
  //   temporary object whose lifetime has not ended, initialized with a
  //   constant expression;
  constexpr volatile S f() { return S(); }
  static_assert(f().i, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
  static_assert(((volatile const S&&)(S)0).i, ""); // expected-error {{constant expression}} expected-note {{read of volatile-qualified type}}
}

// DR1312: The proposed wording for this defect has issues, so we ignore this
// bullet and instead prohibit casts from pointers to cv void (see core-20842
// and core-20845).
//
// - an lvalue-to-rvalue conversion (4.1 [conv.lval]) that is applied to a
// glvalue of type cv1 T that refers to an object of type cv2 U, where T and U
// are neither the same type nor similar types (4.4 [conv.qual]);

// - an lvalue-to-rvalue conversion (4.1) that is applied to a glvalue that
// refers to a non-active member of a union or a subobject thereof;
namespace LValueToRValueUnion {
  // test/SemaCXX/constant-expression-cxx11.cpp contains more thorough testing
  // of this.
  union U { int a, b; } constexpr u = U();
  static_assert(u.a == 0, "");
  constexpr const int *bp = &u.b;
  constexpr int b = *bp; // expected-error {{constant expression}} expected-note {{read of member 'b' of union with active member 'a'}}

  extern const U pu;
  constexpr const int *pua = &pu.a;
  constexpr const int *pub = &pu.b;
  constexpr U pu = { .b = 1 }; // cxx11-warning {{C++20 extension}}
  constexpr const int a2 = *pua; // expected-error {{constant expression}} expected-note {{read of member 'a' of union with active member 'b'}}
  constexpr const int b2 = *pub; // ok
}

// - an id-expression that refers to a variable or data member of reference type
//   unless the reference has a preceding initialization, initialized with a
//   constant expression;
namespace References {
  const int a = 2;
  int &b = *const_cast<int*>(&a);
  int c = 10; // expected-note 2 {{here}}
  int &d = c;
  constexpr int e = 42;
  int &f = const_cast<int&>(e);
  extern int &g; // expected-note {{here}}
  constexpr int &h(); // expected-note {{here}}
  int &i = h(); // expected-note {{here}}
  constexpr int &j() { return b; }
  int &k = j();

  struct S {
    int A : a;
    int B : b;
    int C : c; // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}
    int D : d; // expected-error {{constant expression}} expected-note {{read of non-const variable 'c'}}
    int D2 : &d - &c + 1;
    int E : e / 2;
    int F : f - 11;
    int G : g; // expected-error {{constant expression}} expected-note {{initializer of 'g' is unknown}}
    int H : h(); // expected-error {{constant expression}} expected-note {{undefined function 'h'}}
    int I : i; // expected-error {{constant expression}} expected-note {{initializer of 'i' is not a constant expression}}
    int J : j();
    int K : k;
  };
}

// - a dynamic_cast (5.2.7);
namespace DynamicCast {
  struct S { int n; };
  constexpr S s { 16 };
  struct T {
    int n : dynamic_cast<const S*>(&s)->n; // cxx11-warning {{constant expression}} cxx11-note {{dynamic_cast}}
  };
}

// - a reinterpret_cast (5.2.10);
namespace ReinterpretCast {
  struct S { int n; };
  constexpr S s { 16 };
  struct T {
    int n : reinterpret_cast<const S*>(&s)->n; // expected-warning {{constant expression}} expected-note {{reinterpret_cast}}
  };
  struct U {
    int m : (long)(S*)6; // expected-warning {{constant expression}} expected-note {{reinterpret_cast}}
  };
}

// - a pseudo-destructor call (5.2.4);
namespace PseudoDtor {
  int k;
  typedef int I;
  struct T {
    int n : (k.~I(), 1); // expected-error {{constant expression}} expected-note {{visible outside that expression}}
  };

  constexpr int f(int a = 1) { // cxx11-error {{constant expression}} expected-note {{destroying object 'a' whose lifetime has already ended}}
    return (
        a.~I(), // cxx11-note {{pseudo-destructor}}
        0);
  }
  static_assert(f() == 0, ""); // expected-error {{constant expression}}

  // This is OK in C++20: the union has no active member after the
  // pseudo-destructor call, so the union destructor has no effect.
  union U { int x; };
  constexpr int g(U u = {1}) { // cxx11-error {{constant expression}}
    return (
        u.x.~I(), // cxx11-note 2{{pseudo-destructor}}
        0);
  }
  static_assert(g() == 0, ""); // cxx11-error {{constant expression}} cxx11-note {{in call}}
}

// - increment or decrement operations (5.2.6, 5.3.2);
namespace IncDec {
  int k = 2;
  struct T {
    int n : ++k; // expected-error {{constant expression}} cxx20-note {{visible outside}}
    int m : --k; // expected-error {{constant expression}} cxx20-note {{visible outside}}
  };
}

// - a typeid expression (5.2.8) whose operand is of a polymorphic class type;
namespace std {
  struct type_info {
    virtual ~type_info();
    const char *name;
  };
}
namespace TypeId {
  struct S { virtual void f(); };
  constexpr S *p = 0;
  constexpr const std::type_info &ti1 = typeid(*p); // expected-error {{must be initialized by a constant expression}} cxx11-note {{typeid applied to expression of polymorphic type 'TypeId::S'}} cxx20-note {{dereferenced null pointer}}

  struct T {} t;
  constexpr const std::type_info &ti2 = typeid(t);
}

// - a new-expression (5.3.4);
// - a delete-expression (5.3.5);
namespace NewDelete {
  constexpr int *p = 0;
  struct T {
    int n : *new int(4); // expected-warning {{constant expression}} cxx11-note {{until C++20}} cxx20-note {{was not deallocated}}
    int m : (delete p, 2); // cxx11-warning {{constant expression}} cxx11-note {{until C++20}}
  };
}

// - a relational (5.9) or equality (5.10) operator where the result is
//   unspecified;
namespace UnspecifiedRelations {
  int a, b;
  constexpr int *p = &a, *q = &b;
  // C++11 [expr.rel]p2: If two pointers p and q of the same type point to
  // different objects that are not members of the same array or to different
  // functions, or if only one of them is null, the results of p<q, p>q, p<=q,
  // and p>=q are unspecified.
  constexpr bool u1 = p < q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u2 = p > q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u3 = p <= q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u4 = p >= q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u5 = p < (int*)0; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u6 = p <= (int*)0; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u7 = p > (int*)0; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u8 = p >= (int*)0; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u9 = (int*)0 < q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u10 = (int*)0 <= q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u11 = (int*)0 > q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool u12 = (int*)0 >= q; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  void f(), g();

  constexpr void (*pf)() = &f, (*pg)() = &g;
  constexpr bool u13 = pf < pg; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
                                // expected-warning@-1 {{ordered comparison of function pointers}}
  constexpr bool u14 = pf == pg;

  // If two pointers point to non-static data members of the same object with
  // different access control, the result is unspecified.
  struct A {
  public:
    constexpr A() : a(0), b(0) {}
    int a;
    constexpr bool cmp() const { return &a < &b; } // expected-note {{comparison of address of fields 'a' and 'b' of 'A' with differing access specifiers (public vs private) has unspecified value}}
  private:
    int b;
  };
  static_assert(A().cmp(), ""); // expected-error {{constant expression}} expected-note {{in call}}
  class B {
  public:
    A a;
    constexpr bool cmp() const { return &a.a < &b.a; } // expected-note {{comparison of address of fields 'a' and 'b' of 'B' with differing access specifiers (public vs protected) has unspecified value}}
  protected:
    A b;
  };
  static_assert(B().cmp(), ""); // expected-error {{constant expression}} expected-note {{in call}}

  // If two pointers point to different base sub-objects of the same object, or
  // one points to a base subobject and the other points to a member, the result
  // of the comparison is unspecified. This is not explicitly called out by
  // [expr.rel]p2, but is covered by 'Other pointer comparisons are
  // unspecified'.
  struct C {
    int c[2];
  };
  struct D {
    int d;
  };
  struct E : C, D {
    struct Inner {
      int f;
    } e;
  } e;
  constexpr bool base1 = &e.c[0] < &e.d; // expected-error {{constant expression}} expected-note {{comparison of addresses of subobjects of different base classes has unspecified value}}
  constexpr bool base2 = &e.c[1] < &e.e.f; // expected-error {{constant expression}} expected-note {{comparison of address of base class subobject 'C' of class 'E' to field 'e' has unspecified value}}
  constexpr bool base3 = &e.e.f < &e.d; // expected-error {{constant expression}} expected-note {{comparison of address of base class subobject 'D' of class 'E' to field 'e' has unspecified value}}

  // [expr.rel]p3: Pointers to void can be compared [...] if both pointers
  // represent the same address or are both the null pointer [...]; otherwise
  // the result is unspecified.
  struct S { int a, b; } s;
  constexpr void *null = 0;
  constexpr void *pv = (void*)&s.a;
  constexpr void *qv = (void*)&s.b;
  constexpr bool v1 = null < (int*)0;
  constexpr bool v2 = null < pv; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool v3 = null == pv; // ok
  constexpr bool v4 = qv == pv; // ok
  constexpr bool v5 = qv >= pv; // expected-error {{constant expression}} expected-note {{unequal pointers to void}}
  constexpr bool v6 = qv > null; // expected-error {{constant expression}} expected-note {{comparison has unspecified value}}
  constexpr bool v7 = qv <= (void*)&s.b; // ok
  constexpr bool v8 = qv > (void*)&s.a; // expected-error {{constant expression}} expected-note {{unequal pointers to void}}
}

// - an assignment or a compound assignment (5.17); or
namespace Assignment {
  int k;
  struct T {
    int n : (k = 9); // expected-error {{constant expression}} cxx20-note {{visible outside}}
    int m : (k *= 2); // expected-error {{constant expression}} cxx20-note {{visible outside}}
  };

  struct Literal {
    constexpr Literal(const char *name) : name(name) {}
    const char *name;
  };
  struct Expr {
    constexpr Expr(Literal l) : IsLiteral(true), l(l) {}
    bool IsLiteral;
    union {
      Literal l;
      // ...
    };
  };
  struct MulEq {
    constexpr MulEq(Expr a, Expr b) : LHS(a), RHS(b) {}
    Expr LHS;
    Expr RHS;
  };
  constexpr MulEq operator*=(Expr a, Expr b) { return MulEq(a, b); }
  Literal a("a");
  Literal b("b");
  MulEq c = a *= b; // ok
}

// - a throw-expression (15.1)
namespace Throw {
  struct S {
    int n : (throw "hello", 10); // expected-error {{constant expression}}
  };
}

// PR9999
template<unsigned int v>
class bitWidthHolding {
public:
  static const
  unsigned int width = (v == 0 ? 0 : bitWidthHolding<(v >> 1)>::width + 1);
};

static const int width=bitWidthHolding<255>::width;

template<bool b>
struct always_false {
  static const bool value = false;
};

template<bool b>
struct and_or {
  static const bool and_value = b && and_or<always_false<b>::value>::and_value;
  static const bool or_value = !b || and_or<always_false<b>::value>::or_value;
};

static const bool and_value = and_or<true>::and_value;
static const bool or_value = and_or<true>::or_value;

static_assert(and_value == false, "");
static_assert(or_value == true, "");

namespace rdar13090123 {
  typedef __INTPTR_TYPE__ intptr_t;

  constexpr intptr_t f(intptr_t x) {
    return (((x) >> 21) * 8);
  }

  extern "C" int foo;

  constexpr intptr_t i = f((intptr_t)&foo - 10); // expected-error{{constexpr variable 'i' must be initialized by a constant expression}} \
  // expected-note{{reinterpret_cast}}
}
