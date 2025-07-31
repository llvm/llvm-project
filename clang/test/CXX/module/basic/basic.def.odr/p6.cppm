// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -verify %t/global-vs-module.cppm 
// RUN: %clang_cc1 -std=c++20 -verify %t/global-vs-module-export.cppm
// RUN: %clang_cc1 -std=c++20 -verify %t/global-vs-module-using.cppm
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/M.cppm -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -verify %t/module-vs-global.cpp -fmodule-file=M=%t/M.pcm
//
// Some of the following tests intentionally have no -verify in their RUN
// lines; we are testing that those cases do not produce errors.
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-interface.cpp -fmodule-file=M=%t/M.pcm -verify
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-interface.cpp -fmodule-file=M=%t/M.pcm -DNO_IMPORT
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-interface.cpp -fmodule-file=M=%t/M.pcm -emit-module-interface -o %t/N.pcm -DNO_ERRORS
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-impl.cpp -fmodule-file=M=%t/M.pcm -fmodule-file=N=%t/N.pcm -verify
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-impl.cpp -fmodule-file=M=%t/M.pcm -fmodule-file=N=%t/N.pcm -DNO_IMPORT -verify
//
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-interface.cpp -fmodule-file=M=%t/M.pcm -emit-module-interface -o %t/N-no-M.pcm -DNO_ERRORS -DNO_IMPORT
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-impl.cpp -fmodule-file=M=%t/M.pcm -fmodule-file=N=%t/N-no-M.pcm -verify
// RUN: %clang_cc1 -std=c++20 %t/module-vs-module-impl.cpp -fmodule-file=N=%t/N-no-M.pcm -DNO_IMPORT

//--- global-vs-module.cppm
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

export module M;

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

//--- global-vs-module-export.cppm
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

export module M;

export {
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
}

//--- global-vs-module-using.cppm
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

export module M;

using ::var;
using ::func;
using ::str;
using ::type;
using ::var_tpl;
using ::func_tpl;
using ::str_tpl;
using ::type_tpl;

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

//--- M.cppm
export module M;

export {
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
}

//--- module-vs-global.cpp
module;
import M;

extern int var; // expected-error {{declaration of 'var' in the global module follows declaration in module M}} expected-note@M.cppm:4 {{previous}}
int func(); // expected-error {{declaration of 'func' in the global module follows declaration in module M}} expected-note@M.cppm:5 {{previous}}
struct str; // expected-error {{declaration of 'str' in the global module follows declaration in module M}} expected-note@M.cppm:6 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in the global module follows declaration in module M}} expected-note@M.cppm:9 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in the global module follows declaration in module M}} expected-note@M.cppm:10 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in the global module follows declaration in module M}} expected-note@M.cppm:11 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in the global module follows declaration in module M}} expected-note@M.cppm:12 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;

export module N;

//--- module-vs-module-interface.cpp
export module N;

#ifndef NO_IMPORT
import M;
#endif

#ifndef NO_ERRORS
extern int var; // expected-error {{declaration of 'var' in module N follows declaration in module M}} expected-note@M.cppm:4 {{previous}}
int func(); // expected-error {{declaration of 'func' in module N follows declaration in module M}} expected-note@M.cppm:5 {{previous}}
struct str; // expected-error {{declaration of 'str' in module N follows declaration in module M}} expected-note@M.cppm:6 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in module N follows declaration in module M}} expected-note@M.cppm:9 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in module N follows declaration in module M}} expected-note@M.cppm:10 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in module N follows declaration in module M}} expected-note@M.cppm:11 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in module N follows declaration in module M}} expected-note@M.cppm:12 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;
#endif

//--- module-vs-module-impl.cpp
module N;

#ifndef NO_IMPORT
import M;
#endif

#ifndef NO_ERRORS
extern int var; // expected-error {{declaration of 'var' in module N follows declaration in module M}} expected-note@M.cppm:4 {{previous}}
int func(); // expected-error {{declaration of 'func' in module N follows declaration in module M}} expected-note@M.cppm:5 {{previous}}
struct str; // expected-error {{declaration of 'str' in module N follows declaration in module M}} expected-note@M.cppm:6 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in module N follows declaration in module M}} expected-note@M.cppm:9 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in module N follows declaration in module M}} expected-note@M.cppm:10 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in module N follows declaration in module M}} expected-note@M.cppm:11 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in module N follows declaration in module M}} expected-note@M.cppm:12 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;
#endif
