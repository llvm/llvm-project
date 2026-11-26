// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void nondecl(int (*f)(int x = 5)) // expected-error {{default arguments can only be specified}}
{
  void (*f2)(int = 17)  // expected-error {{default arguments can only be specified}}
  = (void (*)(int = 42))f; // expected-error {{default arguments can only be specified}}
}

struct X0 {
  int (*f)(int = 17); // expected-error{{default arguments can only be specified for parameters in a function declaration}}
  void (*g())(int = 22); // expected-error{{default arguments can only be specified for parameters in a function declaration}}
  void (*h(int = 49))(int);
  auto i(int) -> void (*)(int = 9); // expected-error{{default arguments can only be specified for parameters in a function declaration}}
  
  void mem8(int (*fp)(int) = (int (*)(int = 17))0); // expected-error{{default arguments can only be specified for parameters in a function declaration}}  
};

template <typename... Ts>
void defaultpack(Ts... = 0) {} // expected-error{{parameter pack cannot have a default argument}}

// A lambda parameter pack whose default argument is a pack expansion
// referencing the enclosing function's parameter pack must be diagnosed
// without crashing.
template <class... Types> void lambda_pack_default_arg(Types... args) {
  // FIXME: do not produce these diagnostics. The '...' is the parameter
  // pack's own ellipsis, not an ambiguous C-style varargs ellipsis, so the
  // -Wambiguous-ellipsis warning and its notes should not be emitted here.
  auto lm = [](Types... = args...) {}; // expected-error{{parameter pack cannot have a default argument}} \
                                       // expected-warning{{'...' in this location creates a C-style varargs function}} \
                                       // expected-note{{preceding '...' declares a function parameter pack}} \
                                       // expected-note{{insert ',' before '...' to silence this warning}}
}
