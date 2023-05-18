// RUN: not %clang_cc1 -x c++-module-map %s -emit-module -fmodule-name=Foo -o %t 2>&1 | FileCheck %s
// CHECK: module compilation requires '-fmodules'
module Foo {}
#pragma clang module contents
