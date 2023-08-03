// RUN: c-index-test core -scan-deps %S -- clang_tool %s -I %S/Inputs | FileCheck %s

#include "header.h"

// CHECK: file-deps:
// CHECK-NEXT: flags-c-api.cpp
// CHECK-NEXT: Inputs/header.h
