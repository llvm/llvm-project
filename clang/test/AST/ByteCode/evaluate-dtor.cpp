// RUN: %clang_cc1 -std=c++23 -verify=both,expected %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++23 -verify=both,ref      %s

struct MutableConst {
  struct HasConstMember {
    const int n = 4;
  };
  mutable HasConstMember hcm;
  constexpr ~MutableConst() {
    /// This is _not_ a read.
    auto *p = &hcm.n;
  }
};
constexpr MutableConst mc;


struct  C {
  static int m;
  bool b;
  constexpr C(bool b) : b(b){
  }
  constexpr ~C() {
    if (b)
      m++; // both-note {{a constant expression cannot modify an object that is visible outside that expression}}
  }
};

int C::m = 0;
constexpr C c1(false);
constexpr C c2(true); // both-error {{must have constant destruction}} \
                      // both-note {{in call to}}



namespace LocalVariables {
  struct S {
    constexpr ~S() {
      int a = 0;
      ++a;
    }
  };
  constexpr S s;
}

namespace GlobalVariables {
  struct S {};
  constexpr S s;

  struct K {
    constexpr ~K() {
      s.~S(); // both-note {{a constant expression cannot modify an object that is visible outside that expression}}
    }
  };
  constexpr K k{}; // both-error {{must have constant destruction}} \
                   // both-note {{in call to}}
}
