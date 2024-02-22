// RUN: c-index-test core -scan-deps -working-dir %S -- clang_tool %s -I %S/Inputs | FileCheck %s

#include "header.h"

// CHECK: file-deps:
// CHECK-NEXT: simple-c-api.cpp
// CHECK-NEXT: Inputs/header.h
