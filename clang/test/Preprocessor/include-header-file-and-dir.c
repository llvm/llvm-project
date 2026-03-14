// RUN: %clang_cc1 -E -P %s -I%S/Inputs/include-file-and-dir -I%S/Inputs/include-file-and-dir/incdir -o - | FileCheck %s
#include "file-and-dir/foo.h"

// CHECK: included_foo_dot_h
