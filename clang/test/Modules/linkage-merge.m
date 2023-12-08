// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=linkage_merge_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -w %s -verify

// Test redeclarations of functions where the original declaration is
// still hidden.

@import linkage_merge_left; // excludes "sub"

extern int f0(float);
// expected-error@-1{{conflicting types for 'f0'}}
// expected-note@Inputs/linkage-merge-sub.h:1{{previous declaration}}

static int f1(float); // okay: considered distinct
static int f2(float); // okay: considered distinct
extern int f3(float); // okay: considered distinct

extern float v0;
// expected-error@-1{{redeclaration of 'v0' with a different type: 'float' vs 'int'}}
// expected-note@Inputs/linkage-merge-sub.h:6{{previous declaration is here}}

static float v1;
static float v2;
extern float v3;

typedef float T0;
