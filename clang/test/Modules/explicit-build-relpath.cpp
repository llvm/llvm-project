// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t

// ----------------------
// Build modules A and B.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-module %S/Inputs/explicit-build/module.modulemap -o a.pcm
//
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=a.pcm \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o b-rel.pcm
//
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o b-abs.pcm

// ------------------------------------------
// Mix and match relative and absolute paths.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=a.pcm \
// RUN:            -verify %s
//
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=b-rel.pcm \
// RUN:            -verify %s
//
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=a.pcm \
// RUN:            -fmodule-file=b-abs.pcm \
// RUN:            -verify %s
//
// RUN: not %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=b-rel.pcm \
// RUN:            -fmodule-file=b-abs.pcm \
// RUN:            -verify %s 2>&1 | FileCheck %s
// CHECK: module 'b' is defined in both '{{.*}}b-rel.pcm' and '{{.*}}b-abs.pcm'

#include "a.h"
static_assert(a == 1, "");
// expected-no-diagnostics
