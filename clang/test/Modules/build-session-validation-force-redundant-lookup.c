// RUN: rm -rf %t
// RUN: split-file %s %t

// Verify a module that is built in the same session was looked up during a relocation 
// check when forced. 

// RUN: touch %t/session.timestamp
// RUN: %clang -fmodules -fimplicit-module-maps -fsyntax-only %t/tu1.c \
// RUN:   -fmodules-cache-path=%t/cache -I%t/include \
// RUN:   -fbuild-session-file=%t/session.timestamp -fmodules-validate-once-per-build-session \
// RUN:   -Xclang -fmodules-force-redundant-lookup -Rmodule-validation 2>&1 | FileCheck %s

// CHECK: checking if module 'Dep' from '{{.*}}Dep-{{.*}}.pcm' has relocated

//--- include/module.modulemap
module Dep { header "Dep.h" }
//--- include/Dep.h
int foo(void);

//--- tu1.c
#include "Dep.h"
