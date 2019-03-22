// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-format=obj -fmodule-name=MainA \
// RUN:     -debug-info-kind=limited -dwarf-ext-refs \
// RUN:     -fimplicit-module-maps -x c -fmodules-cache-path=%t -F %S/Inputs \
// RUN:     %s -S -emit-llvm -debugger-tuning=lldb -o - | FileCheck %s

#include "MainA/MainPriv.h"

// CHECK: !DICompileUnit
// CHECK-NOT: dwoId:

// We still want the import, but no skeleton CU, since no PCM was built.

// CHECK: !DIModule({{.*}}, name: "APriv"
// CHECK-NOT: !DICompileUnit
// CHECK-NOT: dwoId:
