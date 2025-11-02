// RUN: %clang_cc1 -fsyntax-only -verify=expected,c23 -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,c17 -std=c17 %s

#define AUTO_MACRO(_NAME, ARG, ARG2, ARG3) \
auto _NAME = ARG + (ARG2 / ARG3);

struct S {
  int a;
  auto b;       // c23-error {{'auto' not allowed in struct member}} \
                   c17-error {{type name does not allow storage class to be specified}} \
                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
  union {
    char c;
    auto smth;  // c23-error {{'auto' not allowed in union member}} \
                   c17-error {{type name does not allow storage class to be specified}} \
                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
  } u;
};

enum E : auto { // c23-error {{'auto' not allowed here}} \
                   c17-error {{expected a type}} \
                   c17-error {{type name does not allow storage class to be specified}}
  One,
  Two,
  Tree,
};

auto basic_usage(auto auto) {   // c23-error {{'auto' not allowed in function prototype}} \
                                   c23-error {{'auto' not allowed in function return type}} \
                                   c23-error {{cannot combine with previous 'auto' declaration specifier}} \
                                   c17-error {{invalid storage class specifier in function declarator}} \
                                   c17-error {{illegal storage class on function}} \
                                   c17-warning {{duplicate 'auto' declaration specifier}} \
                                   c17-warning {{omitting the parameter name in a function definition is a C23 extension}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  auto = 4;                     // expected-error {{expected identifier or '('}}

  auto a = 4;                   // c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  auto auto aa = 12;            // c23-error {{cannot combine with previous 'auto' declaration specifier}} \
                                   c17-warning {{duplicate 'auto' declaration specifier}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  auto b[4];                    // c23-error {{'auto' not allowed in array declaration}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  auto array[auto];             // expected-error {{expected expression}} \
                                   c23-error {{declaration of variable 'array' with deduced type 'auto' requires an initializer}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  AUTO_MACRO(auto, 1, 2, 3);    // c23-error {{cannot combine with previous 'auto' declaration specifier}} \
                                   expected-error {{expected identifier or '('}} \
                                   c17-warning {{duplicate 'auto' declaration specifier}}

  auto c = (auto)a;             // expected-error {{expected expression}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  auto ci = (auto){12};         // expected-error {{expected expression}} \
                                   c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  int auto_cxx_decl = auto(0);  // expected-error {{expected expression}}

  return c;
}

void structs(void) {
  struct s_auto { auto a; };            // c23-error {{'auto' not allowed in struct member}} \
                                           c17-error {{type name does not allow storage class to be specified}} \
                                           c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  // FIXME: this should end up being rejected when we implement underspecified
  // declarations in N3006.
  auto s_int = (struct { int a; } *)0;  // c17-error {{incompatible pointer to integer conversion initializing 'int' with an expression of type}} \
                                           c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  typedef auto auto_type;               // c23-error {{'auto' not allowed in typedef}} \
                                           c17-error {{cannot combine with previous 'typedef' declaration specifier}} \
                                           c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
}

void sizeof_alignas(void) {
  auto auto_size = sizeof(auto);  // expected-error {{expected expression}} \
                                     c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
}

void generic_alignof_alignas(void) {
  int g;
  _Generic(g, auto : 0);  // c23-error {{'auto' not allowed here}} \
                             c17-error {{expected a type}} \
                             c17-error {{type name does not allow storage class to be specified}}

  _Alignof(auto);         // expected-error {{expected expression}} \
                             expected-warning {{'_Alignof' applied to an expression is a GNU extension}}

  _Alignas(auto);         // expected-error {{expected expression}} \
                             expected-warning {{declaration does not declare anything}}
}

void function_designators(void) {
  extern auto auto_ret_func(void);    // c23-error {{'auto' not allowed in function return type}} \
                                         c17-error {{cannot combine with previous 'extern' declaration specifier}} \
                                         c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  extern void auto_param_func(auto);  // c23-error {{'auto' not allowed in function prototype}} \
                                         c17-error {{invalid storage class specifier in function declarator}} \
                                         c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  auto (auto_ret_func)(void);         // c23-error {{'auto' not allowed in function return type}} \
                                         c17-error {{illegal storage class on function}} \
                                         c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  void (auto_param_func)(auto);       // c23-error {{'auto' not allowed in function prototype}} \
                                         c17-error {{invalid storage class specifier in function declarator}} \
                                         c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
}

void atomic(void) {
  _Atomic(auto) atom1 = 12; // c23-error {{'auto' not allowed here}} \
                               c23-error {{a type specifier is required for all declarations}} \
                               c17-error {{expected a type}} \
                               c17-error {{type name does not allow storage class to be specified}} \
                               c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

  _Atomic auto atom2 = 12;  // c23-error {{_Atomic cannot be applied to type 'auto' in C23}} \
                               c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
}

void attributes(void) {
  auto ident [[clang::annotate("this works")]] = 12;  // c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
}

/** GH163090 */
constexpr auto int a1 = 0; // c23-error {{illegal storage class on file-scoped variable}} \
                              c23-error {{cannot combine with previous 'auto' declaration specifier}} \
                              c17-error {{illegal storage class on file-scoped variable}} \
                              c17-error {{unknown type name 'constexpr'}}

constexpr int auto a2 = 0; // c23-error {{cannot combine with previous 'int' declaration specifier}} \
                              c17-error {{illegal storage class on file-scoped variable}} \
                              c17-error {{unknown type name 'constexpr'}}

auto int b1 = 0; // c23-error {{illegal storage class on file-scoped variable}} \
                    c17-error {{illegal storage class on file-scoped variable}}

int auto b2 = 0; // c23-error {{cannot combine with previous 'int' declaration specifier}} \
                    c17-error {{illegal storage class on file-scoped variable}}

void f() {
  constexpr auto int c1 = 0; // c23-error {{cannot combine with previous 'auto' declaration specifier}} \
                                c17-error {{use of undeclared identifier 'constexpr'}}

  constexpr int auto c2 = 0; // c23-error {{cannot combine with previous 'int' declaration specifier}} \
                                c17-error {{use of undeclared identifier 'constexpr'}}

  auto int d1 = 0;
  int auto d2 = 0; // c23-error {{cannot combine with previous 'int' declaration specifier}}
}
