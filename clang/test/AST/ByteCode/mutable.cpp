// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++11 -verify=expected,expected11,both,both11 %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify=expected,expected14,both %s
// RUN: %clang_cc1 -std=c++11 -verify=ref,ref11,both,both11 %s
// RUN: %clang_cc1 -std=c++14 -verify=ref,ref14,both %s





namespace Simple {
  struct S {
    mutable int a; // both-note {{declared here}} \
                   // both11-note {{declared here}}
    int a2;
  };

  constexpr S s{12, 24};
  static_assert(s.a == 12, ""); // both-error {{not an integral constant expression}}  \
                                // both-note {{read of mutable member 'a'}}
  static_assert(s.a2 == 24, "");


  constexpr S s2{12, s2.a}; // both11-error {{must be initialized by a constant expression}} \
                            // both11-note {{read of mutable member 'a'}} \
                            // both11-note {{declared here}}
  static_assert(s2.a2 == 12, ""); // both11-error {{not an integral constant expression}} \
                                  // both11-note {{initializer of 's2' is not a constant expression}}
}
