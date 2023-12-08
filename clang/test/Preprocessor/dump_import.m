// RUN: %clang_cc1 -E -dI %s -o - | FileCheck %s
// CHECK: {{^}}#import "dump_

// See also `dump_include.c` which tests other inclusion cases with `-dI`.

#import "dump_import.h"
