// RUN: rm -rf %t
// Use -DA=0 so that there is at least one preprocessor option serialized after the diagnostic options.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs %s -DA=0 -Rmodule-build -verify
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs %s -DA=0 -Werror -Rmodule-build -verify

@import category_top; // expected-remark {{building module}} expected-remark {{finished building}}
