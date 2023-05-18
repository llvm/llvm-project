// RUN: rm -rf %t
// RUN: %clang_cc1 -I %S/Inputs/getSourceDescriptor-crash -S -emit-llvm -debug-info-kind=limited -debugger-tuning=lldb -fmodules -fmodules-cache-path=%t -fimplicit-module-maps %s -o - | FileCheck %s

#include "h1.h"
#include "h1.h"

// CHECK: DIImportedEntity
// CHECK-SAME: entity: ![[ENTITY:[0-9]+]]
// CHECK: ![[ENTITY]] = !DIModule
// CHECK-SAME: name: "foo"
