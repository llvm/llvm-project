// RUN: rm -rf %t
// RUN: cp -r %S/Inputs/remarks %t
// RUN: cp %s %t/t.cpp

// RUN: clang-tidy -checks='-*,modernize-use-override,clang-diagnostic-module-import' t.cpp -- \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:     -fsyntax-only \
// RUN:     -I%S/Inputs/remarks \
// RUN:     -working-directory=%t \
// RUN:     -Rmodule-build -Rmodule-import t.cpp 2>&1 |\
// RUN: FileCheck %s -implicit-check-not "remark:"

#include "A.h"
// CHECK: remark: importing module 'A' from {{.*}} [clang-diagnostic-module-import]

