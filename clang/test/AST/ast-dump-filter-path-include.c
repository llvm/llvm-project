// RUN: %clang_cc1 -ast-dump -ast-dump-filter-path %s %s | FileCheck %s

#include "filter-header.h"

MAKE_VAR(z)

// CHECK: VarDecl {{.*}} z
