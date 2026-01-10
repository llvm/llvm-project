// RUN: %clang_cc1 -triple x86_64 -fcxx-exceptions -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -triple x86_64 -fcxx-exceptions -std=c++20                                         -verify=ref,both %s

namespace Throw {

  constexpr int ConditionalThrow(bool t) {
    if (t)
      throw 4; // both-note {{subexpression not valid in a constant expression}}

    return 0;
  }

  static_assert(ConditionalThrow(false) == 0, "");
  static_assert(ConditionalThrow(true) == 0, ""); // both-error {{not an integral constant expression}} \
                                                  // both-note {{in call to 'ConditionalThrow(true)'}}

  constexpr int Throw() { // both-error {{never produces a constant expression}}
    throw 5; // both-note {{subexpression not valid in a constant expression}}
    return 0;
  }

  constexpr int NoSubExpr() { // both-error {{never produces a constant expression}}
    throw; // both-note 2{{subexpression not valid}}
    return 0;
  }
  static_assert(NoSubExpr() == 0, ""); // both-error {{not an integral constant expression}} \
                                       // both-note {{in call to}}
}

namespace Asm {
  constexpr int ConditionalAsm(bool t) {
    if (t)
      asm(""); // both-note {{subexpression not valid in a constant expression}}

    return 0;
  }
  static_assert(ConditionalAsm(false) == 0, "");
  static_assert(ConditionalAsm(true) == 0, ""); // both-error {{not an integral constant expression}} \
                                                // both-note {{in call to 'ConditionalAsm(true)'}}


  constexpr int Asm() { // both-error {{never produces a constant expression}}
    __asm volatile(""); // both-note {{subexpression not valid in a constant expression}}
    return 0;
  }
}

namespace Casts {
  constexpr int a = reinterpret_cast<int>(12); // both-error {{must be initialized by a constant expression}} \
                                               // both-note {{reinterpret_cast is not allowed}}

  void func() {
    struct B {};
    B b;
    (void)*reinterpret_cast<void*>(&b); // both-error {{indirection not permitted on operand of type 'void *'}}
  }

  /// Just make sure this doesn't crash.
  float PR9558 = reinterpret_cast<const float&>("asd");
}


/// This used to crash in collectBlock().
struct S {
};
S s;
S *sp[2] = {&s, &s};
S *&spp = sp[1];

namespace InvalidBitCast {
  void foo() {
    const long long int i = 1; // both-note {{declared const here}}
    if (*(double *)&i == 2) {
      i = 0; // both-error {{cannot assign to variable}}
    }
  }

  struct S2 {
    void *p;
  };
  struct T {
    S2 s;
  };
  constexpr T t = {{nullptr}};
  constexpr void *foo2() { return ((void **)&t)[0]; } // both-error {{never produces a constant expression}} \
                                                      // both-note 2{{cast that performs the conversions of a reinterpret_cast}}
  constexpr auto x = foo2(); // both-error {{must be initialized by a constant expression}} \
                             // both-note {{in call to}}


  struct sockaddr
  {
    char sa_data[8];
  };
  struct in_addr
  {
    unsigned int s_addr;
  };
  struct sockaddr_in
  {
    unsigned short int sin_port;
    struct in_addr sin_addr;
  };
  /// Bitcast from sockaddr to sockaddr_in. Used to crash.
  unsigned int get_addr(sockaddr addr) {
      return ((sockaddr_in *)&addr)->sin_addr.s_addr;
  }


  struct s { int a; int b[1]; };
  struct s myx;
  int *myy = ((struct s *)&myx.a)->b;
}

namespace InvalidIntPtrRecord {
  typedef __SIZE_TYPE__ Size_t;

#define bufsize ((1LL << (8 * sizeof(Size_t) - 2)) - 256)

  struct S {
    short buf[bufsize]; // both-error {{array is too large}}
    int a;
  };
  Size_t foo() { return (Size_t)(&((struct S *)0)->a); }
}

namespace RetVoidInInvalidFunc {

  constexpr bool foo() { return; } // both-error {{non-void constexpr function 'foo' should return a value}}
  template <int N> struct X {
    int v = N;
  };
  X<foo()> x; // both-error {{non-type template argument is not a constant expression}}
}
