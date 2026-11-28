// Check that an implicitly built module is out of date when it's validated
// with different -fimplicit-modules-validation-key values than it was built
// with, and that the keys are only compared when the inputs are validated.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: echo "-fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%/t/cache" > %t/ctx.rsp
// RUN: echo "-I %/t/include -Rmodule-build -Rmodule-validation" >> %t/ctx.rsp

// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=a \
// RUN:   2>&1 | FileCheck %s --check-prefix=BUILD
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=a \
// RUN:   2>&1 | FileCheck %s --check-prefix=REUSE --allow-empty
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=b \
// RUN:   2>&1 | FileCheck %s --check-prefixes=A-TO-B,BUILD

// The keys are compared in order.
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=b \
// RUN:   -fimplicit-modules-validation-key=c \
// RUN:   2>&1 | FileCheck %s --check-prefixes=B-TO-BC,BUILD
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=c \
// RUN:   -fimplicit-modules-validation-key=b \
// RUN:   2>&1 | FileCheck %s --check-prefixes=BC-TO-CB,BUILD
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c \
// RUN:   2>&1 | FileCheck %s --check-prefixes=CB-TO-NONE,BUILD

// Once the module was validated in the current build session, the keys aren't
// compared again until the next one.
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=x \
// RUN:   -fmodules-validate-once-per-build-session -fbuild-session-timestamp=1 \
// RUN:   2>&1 | FileCheck %s --check-prefixes=NONE-TO-X,BUILD
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=y \
// RUN:   -fmodules-validate-once-per-build-session -fbuild-session-timestamp=1 \
// RUN:   2>&1 | FileCheck %s --check-prefix=REUSE --allow-empty
// RUN: %clang_cc1 @%t/ctx.rsp %t/tu.c -fimplicit-modules-validation-key=y \
// RUN:   -fmodules-validate-once-per-build-session \
// RUN:   -fbuild-session-timestamp=4000000000 \
// RUN:   2>&1 | FileCheck %s --check-prefixes=X-TO-Y,BUILD

// A precompiled header can't rebuild the modules it imports, so they are used
// even though the keys differ.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%/t/cache \
// RUN:   -I %/t/include -emit-pch -x c-header %t/pch.h -o %t/pch.pch \
// RUN:   -fimplicit-modules-validation-key=p
// RUN: %clang_cc1 @%t/ctx.rsp %t/use-pch.c -include-pch %t/pch.pch \
// RUN:   -fimplicit-modules-validation-key=q \
// RUN:   2>&1 | FileCheck %s --check-prefix=REUSE-NOT-REBUILT --allow-empty

// A-TO-B: remark: module 'Mod' is out of date because it was built with validation keys 'a' but the current keys are 'b'
// B-TO-BC: remark: module 'Mod' is out of date because it was built with validation keys 'b' but the current keys are 'b', 'c'
// BC-TO-CB: remark: module 'Mod' is out of date because it was built with validation keys 'b', 'c' but the current keys are 'c', 'b'
// CB-TO-NONE: remark: module 'Mod' is out of date because it was built with validation keys 'c', 'b' but the current keys are none
// NONE-TO-X: remark: module 'Mod' is out of date because it was built with validation keys none but the current keys are 'x'
// X-TO-Y: remark: module 'Mod' is out of date because it was built with validation keys 'x' but the current keys are 'y'
// BUILD: remark: building module 'Mod'
// REUSE: validating {{[0-9]+}} input files in module 'Mod'
// REUSE-NOT: out of date
// REUSE-NOT: building module
// REUSE-NOT-REBUILT-NOT: out of date
// REUSE-NOT-REBUILT-NOT: building module

//--- include/module.modulemap
module Mod { header "mod.h" }

//--- include/mod.h
int mod;

//--- tu.c
#include "mod.h"

//--- pch.h
#include "mod.h"

//--- use-pch.c
int use(void) { return mod; }
