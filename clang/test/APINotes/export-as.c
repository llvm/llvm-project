// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInt -x c | FileCheck %s

#include "ExportAs.h"

// CHECK: Dumping globalInt:
// CHECK: VarDecl {{.+}} imported in ExportAsCore globalInt 'int'
// CHECK: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"
