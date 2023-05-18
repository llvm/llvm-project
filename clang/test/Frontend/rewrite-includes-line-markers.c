// RUN: %clang_cc1 -E -frewrite-includes -I %S/Inputs %s | FileCheck %s --check-prefix=GNU
// RUN: %clang_cc1 -E -frewrite-includes -fuse-line-directives -I %S/Inputs %s | FileCheck %s --check-prefix=LINE
#include "test.h"
int f() { return x; }

#include "empty.h"

// GNU: {{^}}# 1 "{{.*}}rewrite-includes-line-markers.c"
// GNU: {{^}}#include "test.h"
// GNU: {{^}}# 1 "{{.*}}test.h"
// GNU: {{^}}#include "test2.h"
// GNU: {{^}}# 1 "{{.*}}test2.h"
// GNU: {{^}}int x;
// GNU: {{^}}# 4 "{{.*}}rewrite-includes-line-markers.c" 2
// GNU: {{^}}int f() { return x; }
// GNU: {{^}}
// GNU: {{^}}# 1 "{{.*}}empty.h" 1
// GNU: {{^}}# 7 "{{.*}}rewrite-includes-line-markers.c" 2

// LINE: {{^}}#line 1 "{{.*}}rewrite-includes-line-markers.c"
// LINE: {{^}}#include "test.h"
// LINE: {{^}}#line 1 "{{.*}}test.h"
// LINE: {{^}}#include "test2.h"
// LINE: {{^}}#line 1 "{{.*}}test2.h"
// LINE: {{^}}int x;
// LINE: {{^}}#line 4 "{{.*}}rewrite-includes-line-markers.c"
// LINE: {{^}}int f() { return x; }
// LINE: {{^}}
// LINE: {{^}}#line 1 "{{.*}}empty.h"
// LINE: {{^}}#line 7 "{{.*}}rewrite-includes-line-markers.c"
