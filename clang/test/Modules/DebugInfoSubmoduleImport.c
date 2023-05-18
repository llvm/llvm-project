// XFAIL: target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-format=obj -debug-info-kind=limited -dwarf-ext-refs \
// RUN:     -fimplicit-module-maps -x c -fmodules-cache-path=%t -I %S/Inputs \
// RUN:     %s -emit-llvm -debugger-tuning=lldb -o - | FileCheck %s
//
// RUN: %clang_cc1 -fmodules -fmodule-format=obj -debug-info-kind=limited -dwarf-ext-refs \
// RUN:     -fimplicit-module-maps -x c -fmodules-cache-path=%t -I %S/Inputs \
// RUN:     -fmodules-local-submodule-visibility \
// RUN:     %s -emit-llvm -debugger-tuning=lldb -o - | FileCheck %s
#include "DebugSubmoduleA.h"
#include "DebugSubmoduleB.h"

// CHECK: !DICompileUnit
// CHECK-NOT: !DICompileUnit
// CHECK: !DIModule(scope: ![[PARENT:.*]], name: "DebugSubmoduleA"
// CHECK: [[PARENT]] = !DIModule(scope: null, name: "DebugSubmodules"
// CHECK: !DIModule(scope: ![[PARENT]], name: "DebugSubmoduleB"
// CHECK: !DICompileUnit({{.*}}splitDebugFilename: {{.*}}DebugSubmodules
// CHECK-SAME:                 dwoId:
// CHECK-NOT: !DICompileUnit
