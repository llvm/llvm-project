// RUN: rm -rf %t
//
// For an implicit module, all that matters are the warning flags in the user.
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodules-cache-path=%t -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -Wpadded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -Wpadded -Werror
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DERROR -Werror=padded
//
// For an explicit module, all that matters are the warning flags in the module build.
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/nodiag.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -fmodule-file=%t/nodiag.pcm -Wpadded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -fmodule-file=%t/nodiag.pcm -Werror -Wpadded
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/warning.pcm -Wpadded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/warning.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/warning.pcm -Werror=padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/warning.pcm -Werror
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -emit-module -fmodule-name=diag_flags -x c++ %S/Inputs/module.map -fmodules-ts -o %t/werror-no-error.pcm -Werror -Wpadded -Wno-error=padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/werror-no-error.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/werror-no-error.pcm -Wno-padded
// RUN: %clang_cc1 -triple %itanium_abi_triple -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs %s -fmodules-ts -DWARNING -fmodule-file=%t/werror-no-error.pcm -Werror=padded
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
// expected-error@diag_flags.h:14 {{padding struct}}
#elif WARNING
// expected-warning@diag_flags.h:14 {{padding struct}}
#else
// expected-no-diagnostics
#endif
int arr[sizeof(Padded)];
