// RUN: rm -rf %t
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodules-cache-path=%t -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DIMPLICIT_FLAG -Werror=padded
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/explicit.pcm -Werror=string-plus-int -Wpadded
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DEXPLICIT_FLAG -fmodule-file=%t/explicit.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DEXPLICIT_FLAG -fmodule-file=%t/explicit.pcm -Werror=padded

import diag_flags;

// Diagnostic flags from the module user make no difference to diagnostics
// emitted within the module when using an explicitly-loaded module.
#ifdef IMPLICIT_FLAG
// expected-error@diag_flags.h:14 {{padding struct}}
#elif defined(EXPLICIT_FLAG)
// expected-warning@diag_flags.h:14 {{padding struct}}
#else
// expected-no-diagnostics
#endif
unsigned n = sizeof(Padded);
