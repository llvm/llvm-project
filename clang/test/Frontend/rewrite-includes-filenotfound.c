// RUN: not %clang_cc1 -E -frewrite-includes %s -o - 2>&1 | FileCheck %s

#include "this file does not exist.foo"
#include "this file also does not exist.foo"

CHECK: fatal error: {{.*}} file not found
