// RUN: %clang_cc1 -fmodules-ts -verify -std=c++17 %s 
// RUN: %clang_cc1 -fmodules-ts -verify -std=c++17 %s -DEXPORT
// RUN: %clang_cc1 -fmodules-ts -verify -std=c++17 %s -DUSING

#ifndef NO_GLOBAL
extern int var; // expected-note {{previous declaration is here}}
int func(); // expected-note {{previous declaration is here}}
struct str; // expected-note {{previous declaration is here}}
using type = int;

template<typename> extern int var_tpl; // expected-note {{previous declaration is here}}
template<typename> int func_tpl(); // expected-note-re {{{{previous declaration is here|target of using declaration}}}}
template<typename> struct str_tpl; // expected-note {{previous declaration is here}}
template<typename> using type_tpl = int; // expected-note {{previous declaration is here}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;
#endif

export module M;

#ifdef USING
using ::var;
using ::func;
using ::str;
using ::type;
using ::var_tpl;
using ::func_tpl; // expected-note {{using declaration}}
using ::str_tpl;
using ::type_tpl;
#endif

#ifdef EXPORT
export {
#endif

extern int var; // expected-error {{declaration of 'var' in module M follows declaration in the global module}}
int func(); // expected-error {{declaration of 'func' in module M follows declaration in the global module}}
struct str; // expected-error {{declaration of 'str' in module M follows declaration in the global module}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in module M follows declaration in the global module}}
// FIXME: Is this the right diagnostic in the -DUSING case?
template<typename> int func_tpl(); // expected-error-re {{{{declaration of 'func_tpl' in module M follows declaration in the global module|conflicts with target of using declaration}}}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in module M follows declaration in the global module}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in module M follows declaration in the global module}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;

#ifdef EXPORT
}
#endif
