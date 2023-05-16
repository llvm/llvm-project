// RUN: %clang_cc1 -fsyntax-only -Wno-header-guard %s
// RUN: %clang_cc1 -fsyntax-only -Wheader-guard %s 2>&1 | FileCheck %s

#include "Inputs/good-header-guard.h"
#include "Inputs/no-define.h"
#include "Inputs/different-define.h"
#include "Inputs/out-of-order-define.h"
#include "Inputs/tokens-between-ifndef-and-define.h"
#include "Inputs/unlikely-to-be-header-guard.h"

#include "Inputs/bad-header-guard.h"
// CHECK: In file included from {{.*}}header.cpp:{{[0-9]*}}:
// CHECK: {{.*}}bad-header-guard.h:1:9: warning: 'bad_header_guard' is used as a header guard here, followed by #define of a different macro
// CHECK: {{^}}#ifndef bad_header_guard
// CHECK: {{^}}        ^~~~~~~~~~~~~~~~
// CHECK: {{.*}}bad-header-guard.h:2:9: note: 'bad_guard' is defined here; did you mean 'bad_header_guard'?
// CHECK: {{^}}#define bad_guard
// CHECK: {{^}}        ^~~~~~~~~
// CHECK: {{^}}        bad_header_guard

#include "Inputs/bad-header-guard-defined.h"
// CHECK: In file included from {{.*}}header.cpp:{{[0-9]*}}:
// CHECK: {{.*}}bad-header-guard-defined.h:1:2: warning: 'foo' is used as a header guard here, followed by #define of a different macro
// CHECK: {{^}}#if !defined(foo)
// CHECK: {{^}} ^~
// CHECK: {{.*}}bad-header-guard-defined.h:2:9: note: 'goo' is defined here; did you mean 'foo'?
// CHECK: {{^}}#define goo
// CHECK: {{^}}        ^~~
// CHECK: {{^}}        foo

#include "Inputs/multiple.h"
#include "Inputs/multiple.h"
#include "Inputs/multiple.h"
#include "Inputs/multiple.h"
// CHECK: In file included from {{.*}}header.cpp:{{[0-9]*}}:
// CHECK: {{.*}}multiple.h:1:9: warning: 'multiple' is used as a header guard here, followed by #define of a different macro
// CHECK: {{^}}#ifndef multiple
// CHECK: {{^}}        ^~~~~~~~
// CHECK: {{.*}}multiple.h:2:9: note: 'multi' is defined here; did you mean 'multiple'?
// CHECK: {{^}}#define multi
// CHECK: {{^}}        ^~~~~
// CHECK: {{^}}        multiple

// CHECK: 3 warnings generated.
