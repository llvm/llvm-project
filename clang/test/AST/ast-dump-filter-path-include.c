// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ast-dump -ast-dump-filter-path "*ast-dump-filter-path-include.c" %s | FileCheck %s

#include "filter-header.h"

MAKE_VAR(z)

// CHECK: VarDecl {{.*}} z