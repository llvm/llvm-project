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
