// RUN: %clang_cc1 -std=c++2a -emit-header-module -fmodule-name=attrs -x c++-header %S/Inputs/empty.h %S/Inputs/attrs.h -o %t.pcm
// RUN: %clang_cc1 -std=c++2a %s -fmodule-file=%t.pcm -E -verify -I%S/Inputs | FileCheck %s

#define SEMI ;
// expected-error@+1 {{semicolon terminating header import declaration cannot be produced by a macro}}
import "empty.h" SEMI // CHECK: import attrs.{{.*}};

#define IMPORT import "empty.h"
IMPORT; // CHECK: import attrs.{{.*}};

#define IMPORT_ANGLED import <empty.h>
IMPORT_ANGLED; // CHECK: import attrs.{{.*}};

// Ensure that macros only become visible at the semicolon.
// CHECK: import attrs.{{.*}} ATTRS ;
import "attrs.h" ATTRS ;
// CHECK: {{\[\[}} ]] int n;
ATTRS int n;
