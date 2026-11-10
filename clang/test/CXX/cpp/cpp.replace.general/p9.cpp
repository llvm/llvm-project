// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=default %s
// RUN: %clang_cc1 -std=c++26 -pedantic -fsyntax-only -verify=pedantic %s
// RUN: %clang_cc1 -std=c++26 -pedantic -Wno-keyword-macro-undef -fsyntax-only -verify=default %s
// RUN: %clang_cc1 -std=c++26 -pedantic-errors -fsyntax-only -verify=pedantic-errors %s

// [cpp.replace.general]/p9: A translation unit shall not #define or #undef
// macro names lexically identical to keywords ([lex.key]) or to the identifiers
// listed in Table 4.
#define for 0
// default-error@-1 {{keyword is hidden by macro definition}}
// pedantic-error@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
#undef for
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

#define final
// default-error@-1 {{keyword is hidden by macro definition}}
// pedantic-error@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
struct S final {
// default-note@-1 {{expansion of macro 'final' requested here}}
// pedantic-note@-2 {{expansion of macro 'final' requested here}}
// pedantic-errors-note@-3 {{expansion of macro 'final' requested here}}
  virtual void f();
};
#undef final
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

#define final 1
// default-error@-1 {{keyword is hidden by macro definition}}
// pedantic-error@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
struct S2 final {};
// default-note@-1 {{expansion of macro 'final' requested here}}
// pedantic-note@-2 {{expansion of macro 'final' requested here}}
// pedantic-errors-note@-3 {{expansion of macro 'final' requested here}}
// default-error@-4 {{expected unqualified-id}}
// pedantic-error@-5 {{expected unqualified-id}}
// pedantic-errors-error@-6 {{expected unqualified-id}}
#undef final
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

// Contextual keywords not used in a contextual keyword position do not warn on #define.
#define final 1
int not_contextual_final = final;
#undef final
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

#define override
// default-error@-1 {{keyword is hidden by macro definition}}
// pedantic-error@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
struct D : S {
  void f() override;
// default-note@-1 {{expansion of macro 'override' requested here}}
// pedantic-note@-2 {{expansion of macro 'override' requested here}}
// pedantic-errors-note@-3 {{expansion of macro 'override' requested here}}
};
#undef override
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

#define module 1
int module_var = module;
#undef module
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

// Empty definitions of qualifier keywords are accepted for compatibility with
// configuration scripts, but #undef is still diagnosed in pedantic modes.
#define const
#undef const
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}
