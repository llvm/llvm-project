// RUN: rm -rf %t
// RUN: mkdir %t
//
// Some of the following tests intentionally have no -verify in their RUN
// lines; we are testing that those cases do not produce errors.
//
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %S/global-vs-module.cpp -emit-module-interface -o %t/M.pcm -DNO_GLOBAL -DEXPORT
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/M.pcm -DMODULE_INTERFACE -verify
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/M.pcm -DMODULE_INTERFACE -DNO_IMPORT
//
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/M.pcm -emit-module-interface -o %t/N.pcm -DMODULE_INTERFACE -DNO_ERRORS
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/N.pcm -verify
// FIXME: Once we start importing "import" declarations properly, this should
// be rejected (-verify should be added to the following line).
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/N.pcm -DNO_IMPORT
//
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/M.pcm -emit-module-interface -o %t/N-no-M.pcm -DMODULE_INTERFACE -DNO_ERRORS -DNO_IMPORT
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/N-no-M.pcm -verify
// RUN: %clang_cc1 -fmodules-ts -std=c++17 %s -fmodule-file=%t/N-no-M.pcm -DNO_IMPORT

#ifdef MODULE_INTERFACE
export
#endif
module N;

#ifndef NO_IMPORT
import M;
#endif

#ifndef NO_ERRORS
extern int var; // expected-error {{declaration of 'var' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:38 {{previous}}
int func(); // expected-error {{declaration of 'func' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:39 {{previous}}
struct str; // expected-error {{declaration of 'str' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:40 {{previous}}
using type = int;

template<typename> extern int var_tpl; // expected-error {{declaration of 'var_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:43 {{previous}}
template<typename> int func_tpl(); // expected-error {{declaration of 'func_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:44 {{previous}}
template<typename> struct str_tpl; // expected-error {{declaration of 'str_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:45 {{previous}}
template<typename> using type_tpl = int; // expected-error {{declaration of 'type_tpl' in module N follows declaration in module M}} expected-note@global-vs-module.cpp:46 {{previous}}

typedef int type;
namespace ns { using ::func; }
namespace ns_alias = ns;
#endif
