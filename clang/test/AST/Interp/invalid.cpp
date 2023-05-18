// RUN: %clang_cc1 -fcxx-exceptions -std=c++20 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -std=c++20 -verify=ref %s

namespace Throw {

  constexpr int ConditionalThrow(bool t) {
    if (t)
      throw 4; // expected-note {{subexpression not valid in a constant expression}} \
               // ref-note {{subexpression not valid in a constant expression}}

    return 0;
  }

  static_assert(ConditionalThrow(false) == 0, "");
  static_assert(ConditionalThrow(true) == 0, ""); // expected-error {{not an integral constant expression}} \
                                                  // expected-note {{in call to 'ConditionalThrow(true)'}} \
                                                  // ref-error {{not an integral constant expression}} \
                                                  // ref-note {{in call to 'ConditionalThrow(true)'}}

  constexpr int Throw() { // expected-error {{never produces a constant expression}} \
                          // ref-error {{never produces a constant expression}}
    throw 5; // expected-note {{subexpression not valid in a constant expression}} \
             // ref-note {{subexpression not valid in a constant expression}}
    return 0;
  }

  constexpr int NoSubExpr() { // ref-error {{never produces a constant expression}} \
                              // expected-error {{never produces a constant expression}}
    throw; // ref-note 2{{subexpression not valid}} \
           // expected-note 2{{subexpression not valid}}
    return 0;
  }
  static_assert(NoSubExpr() == 0, ""); // ref-error {{not an integral constant expression}} \
                                       // ref-note {{in call to}} \
                                       // expected-error {{not an integral constant expression}} \
                                       // expected-note {{in call to}}
}

namespace Asm {
  constexpr int ConditionalAsm(bool t) {
    if (t)
      asm(""); // expected-note {{subexpression not valid in a constant expression}} \
               // ref-note {{subexpression not valid in a constant expression}}

    return 0;
  }
  static_assert(ConditionalAsm(false) == 0, "");
  static_assert(ConditionalAsm(true) == 0, ""); // expected-error {{not an integral constant expression}} \
                                                // expected-note {{in call to 'ConditionalAsm(true)'}} \
                                                // ref-error {{not an integral constant expression}} \
                                                // ref-note {{in call to 'ConditionalAsm(true)'}}


  constexpr int Asm() { // expected-error {{never produces a constant expression}} \
                        // ref-error {{never produces a constant expression}}
    __asm volatile(""); // expected-note {{subexpression not valid in a constant expression}} \
                        // ref-note {{subexpression not valid in a constant expression}}
    return 0;
  }
}

namespace Casts {
  constexpr int a = reinterpret_cast<int>(12); // expected-error {{must be initialized by a constant expression}} \
                                               // expected-note {{reinterpret_cast is not allowed}} \
                                               // ref-error {{must be initialized by a constant expression}} \
                                               // ref-note {{reinterpret_cast is not allowed}}

}
