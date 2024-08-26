// RUN: %clang_cc1 -fcxx-exceptions -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -fcxx-exceptions -std=c++20 -verify=ref,both %s

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
}
