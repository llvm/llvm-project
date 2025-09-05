// RUN: %clang_cc1 -std=c++03 -verify=expected,both %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++03 -verify=ref,both %s

namespace NonInitializingMemberExpr {
  struct NonLit {
    NonLit() : value(0) {}
    int value;
  };
  __attribute__((require_constant_initialization)) const int &nl_subobj_ref = NonLit().value; // both-error {{variable does not have a constant initializer}} \
                                                                                              // both-note {{required by}} \
                                                                                              // both-note {{subexpression not valid}}
}


namespace NonLValueMemberExpr {
  struct PODType {
    int value;
  };

#define ATTR __attribute__((require_constant_initialization))
  struct TT1 {
    ATTR static const int &subobj_init;
  };

  const int &TT1::subobj_init = PODType().value;
}

void LambdaAccessingADummy() {
  int d;
  int a9[1] = {[d = 0] = 1}; // both-error {{is not an integral constant expression}}
}

const int p = 10;
struct B {
  int a;
  void *p;
};
struct B2 : B {
  void *q;
};
_Static_assert(&(B2().a) == &p, ""); // both-error {{taking the address of a temporary object of type 'int'}} \
                                     // both-error {{not an integral constant expression}}
