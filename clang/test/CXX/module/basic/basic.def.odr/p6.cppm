// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -verify %t/global-vs-module.cppm 
// RUN: %clang_cc1 -std=c++20 -verify %t/global-vs-module.cppm -DEXPORT
// RUN: %clang_cc1 -std=c++20 -verify %t/global-vs-module.cppm -DUSING
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/global-vs-module.cppm -o %t/M.pcm -DNO_GLOBAL -DEXPORT
// RUN: %clang_cc1 -std=c++20 -verify %t/module-vs-global.cpp -fmodule-file=M=%t/M.pcm
//
// Some of the following tests intentionally have no -verify in their RUN
// lines; we are testing that those cases do not produce errors.
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -DMODULE_INTERFACE -verify
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -DMODULE_INTERFACE -DNO_IMPORT
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -emit-module-interface -o %t/N.pcm -DMODULE_INTERFACE -DNO_ERRORS
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -fmodule-file=N=%t/N.pcm -verify
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -fmodule-file=N=%t/N.pcm -DNO_IMPORT -verify
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -emit-module-interface -o %t/N-no-M.pcm -DMODULE_INTERFACE -DNO_ERRORS -DNO_IMPORT
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=M=%t/M.pcm -fmodule-file=N=%t/N-no-M.pcm -verify
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module.cpp -fmodule-file=N=%t/N-no-M.pcm -DNO_IMPORT

//--- global-vs-module.cppm
#ifndef NO_GLOBAL
module;
extern int var; // expected-note {{previous declaration is here}}
int func(); // expected-note {{previous declaration is here}}
struct str; // expected-note {{previous declaration is here}}
using type = int;

template<typename> extern int var_tpl; // expected-note {{previous declaration is here}}
template<typename> int func_tpl(); // expected-note {{previous declaration is here}}
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
using ::func_tpl;
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
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in module M follows declaration in the global module}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in module M follows declaration in the global module}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in module M follows declaration in the global module}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;

#ifdef EXPORT
}
#endif

//--- module-vs-global.cpp
import M;

extern int var; // expected-error {{declaration of 'var' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:35 {{previous}}
int func(); // expected-error {{declaration of 'func' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:36 {{previous}}
struct str; // expected-error {{declaration of 'str' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:37 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:40 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:41 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:42 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in the global module follows declaration in module M}} expected-note@global-vs-module.cppm:43 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;

//--- module-vs-module.cpp
#ifdef MODULE_INTERFACE
export module N;
#else
module N;
#endif

#ifndef NO_IMPORT
import M;
#endif

#ifndef NO_ERRORS
extern int var; // expected-error {{declaration of 'var' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:35 {{previous}}
int func(); // expected-error {{declaration of 'func' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:36 {{previous}}
struct str; // expected-error {{declaration of 'str' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:37 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:40 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:41 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:42 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cppm:43 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;
#endif

