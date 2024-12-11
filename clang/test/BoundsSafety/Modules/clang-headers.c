
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fmodules \
// RUN:     -fimplicit-module-maps -fmodules-cache-path=%t/module_cache -Rmodule-import -verify %s
// RUN: %clang_cc1 -fsyntax-only                                 -fmodules \
// RUN:     -fimplicit-module-maps -fmodules-cache-path=%t/module_cache -Rmodule-import -verify %s

#include <ptrcheck.h> // expected-remark-re{{importing module 'ptrcheck' from '{{.*}}.pcm'}}
