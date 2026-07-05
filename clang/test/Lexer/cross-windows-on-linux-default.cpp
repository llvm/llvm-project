// RUN: not %clang_cc1 -fsyntax-only -fms-extensions -triple i686-win32 %s 2>&1 \
// RUN:   | FileCheck %s

#include "Inputs\success.h"

// CHECK: error: success
