// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_C
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_C -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DEXTERN_C -DNAMESPACE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C -DEXTERN_CXX
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C -DNAMESPACE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C -DNO_EXTERN_C_ERROR -Wno-module-import-in-extern-c
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -DCXX_HEADER -DEXTERN_C -DNAMESPACE -DNO_EXTERN_C_ERROR -Wno-module-import-in-extern-c
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs -x c %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs/elsewhere -I %S/Inputs %s -DEXTERN_C -DINDIRECT

#ifdef INDIRECT
#include "c-header-indirect.h"
#endif

#ifdef NAMESPACE
namespace M {
#endif

#ifdef EXTERN_C
extern "C" {
#endif

#ifdef EXTERN_CXX
extern "C++" {
#endif

#ifdef CXX_HEADER
#define HEADER "cxx-header.h"
#else
#define HEADER "c-header.h"
#endif

#include HEADER

#if defined(NAMESPACE)
// expected-error-re@-3 {{import of module '{{c_library.inner|cxx_library}}' appears within namespace 'M'}}
// expected-note@-21 {{namespace 'M' begins here}}
#elif defined(EXTERN_C) && !defined(EXTERN_CXX) && defined(CXX_HEADER) && !defined(NO_EXTERN_C_ERROR)
// expected-error@-6 {{import of C++ module 'cxx_library' appears within extern "C" language linkage specification}}
// expected-note@-20 {{extern "C" language linkage specification begins here}}
#endif

#ifdef EXTERN_CXX
}
#endif

#ifdef EXTERN_C
}
#endif

#ifdef NAMESPACE
}
using namespace M;
#endif

#ifdef __cplusplus
namespace N {
#endif
  void g(void) {
    int k = f();
  }

#ifdef __cplusplus
  extern "C" {
#endif
    int f;
#if !defined(CXX_HEADER) && !defined(NAMESPACE)
    // expected-error@-2 {{redefinition of 'f' as different kind of symbol}}
    // expected-note@c-header.h:1 {{previous}}
#endif

#ifdef __cplusplus
  }
}
#endif

#if !defined(NAMESPACE)
suppress_expected_no_diagnostics_error error_here; // expected-error {{}}
#endif
