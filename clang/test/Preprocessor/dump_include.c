// RUN: %clang_cc1 -w -E -dI -isystem %S -imacros %S/dump_include.h %s -o - | FileCheck %s
// CHECK: {{^}}#__include_macros "{{.*}}dump_
// CHECK: {{^}}#include <dump_
// CHECK: {{^}}#include "dump_
// CHECK: {{^}}#include_next "dump_

// See also `dump_import.m` which tests the `#import` directive with `-dI`.

#include <dump_include.h>
#include "dump_include.h"
#include_next "dump_include.h"
