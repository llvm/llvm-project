// RUN: rm -rf %t
//
// For an implicit module, all that matters are the warning flags in the user.
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodules-cache-path=%t -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -DLOCAL_WARNING -Wpadded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -DLOCAL_ERROR -Wpadded -Werror
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -DLOCAL_ERROR -Werror=padded
//
// For an explicit module, all that matters are the warning flags in the module build.
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/nodiag.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -fmodule-file=%t/nodiag.pcm -Wpadded -DLOCAL_WARNING
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -fmodule-file=%t/nodiag.pcm -Werror -Wpadded -DLOCAL_ERROR
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/warning.pcm -Wpadded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/warning.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/warning.pcm -Werror=padded -DLOCAL_ERROR
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/warning.pcm -Werror
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/werror-no-error.pcm -Werror -Wpadded -Wno-error=padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/werror-no-error.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/werror-no-error.pcm -Wno-padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/werror-no-error.pcm -Werror=padded -DLOCAL_ERROR
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/error.pcm -Werror=padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -fmodule-file=%t/error.pcm -Wno-padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -fmodule-file=%t/error.pcm -Wno-error=padded
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/werror.pcm -Werror -Wpadded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -fmodule-file=%t/werror.pcm -Wno-error
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -fmodule-file=%t/werror.pcm -Wno-padded

import diag_flags;

// Diagnostic flags from the module user make no difference to diagnostics
// emitted within the module when using an explicitly-loaded module.
#if ERROR
// expected-error@diag_flags.h:4 {{padding struct}}
#elif WARNING
// expected-warning@diag_flags.h:4 {{padding struct}}
#endif
int arr[sizeof(Padded)];

// Diagnostic flags from the module make no difference to diagnostics emitted
// in the module user.
#if LOCAL_ERROR
// expected-error@+4 {{padding struct}}
#elif LOCAL_WARNING
// expected-warning@+2 {{padding struct}}
#endif
struct Padded2 { char x; int y; };
int arr2[sizeof(Padded2)];

#if !ERROR && !WARNING && !LOCAL_ERROR && !LOCAL_WARNING
// expected-no-diagnostics
#endif
