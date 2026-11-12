// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fmodules -fmodules-codegen -emit-module -fmodule-name=foo %S/Inputs/codegen-extern-template.modulemap -x c++ -o %t.pcm
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fmodules -fmodule-file=%t.pcm %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#include "codegen-extern-template.h"

template int foo<int>();

// CHECK: define weak_odr noundef i32 @_Z3fooIiET_v
