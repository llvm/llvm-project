// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-ts -emit-module-interface -std=c++17 %S/global-vs-module.cpp -o %t -DNO_GLOBAL -DEXPORT
// RUN: %clang_cc1 -fmodules-ts -verify -std=c++17 %s -fmodule-file=%t

import M;

extern int var; // expected-error {{declaration of 'var' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:38 {{previous}}
int func(); // expected-error {{declaration of 'func' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:39 {{previous}}
struct str; // expected-error {{declaration of 'str' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:40 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:43 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:44 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:45 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cpp:46 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;
