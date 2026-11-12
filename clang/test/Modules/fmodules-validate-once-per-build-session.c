// This tests the behavior of -fmodules-validate-once-per-build-session with
// different combinations of flags and states of the module cache.

// Note: The `sleep 1` commands sprinkled throughout this test make the strict
//       comparisons of epoch mtimes work as expected. Some may be unnecessary,
//       but make the intent clearer.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: echo "-fsyntax-only -fmodules -fmodules-cache-path=%/t/module-cache" > %t/ctx.rsp
// RUN: echo "-fbuild-session-file=%/t/module-cache/session.timestamp"      >> %t/ctx.rsp
// RUN: echo "-fmodules-validate-once-per-build-session"                    >> %t/ctx.rsp
// RUN: echo "-Rmodule-build -Rmodule-validation"                           >> %t/ctx.rsp

//--- include/foo.h
//--- include/module.modulemap
module Foo { header "foo.h" }

//--- clean.c
// Clean module cache. Modules will get compiled regardless of validation settings.
// RUN: mkdir %t/module-cache
// RUN: sleep 1
// RUN: touch %t/module-cache/session.timestamp
// RUN: sleep 1
// RUN: %clang @%t/ctx.rsp %t/clean.c -DCTX=1 \
// RUN:   -isystem %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/clean.c
// RUN: %clang @%t/ctx.rsp %t/clean.c -DCTX=2 \
// RUN:   -I %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/clean.c
// RUN: %clang @%t/ctx.rsp %t/clean.c -DCTX=3 \
// RUN:   -I %t/include -fmodules-validate-system-headers -Xclang -fno-modules-force-validate-user-headers \
// RUN:     2>&1 | FileCheck %t/clean.c
#include "foo.h"
// CHECK: building module 'Foo'

//--- no-change-same-session.c
// Populated module cache in the same build session with unchanged inputs.
// Validation only happens when it's forced for user headers. No compiles.
// RUN: sleep 1
// RUN: %clang @%t/ctx.rsp %t/no-change-same-session.c -DCTX=1 \
// RUN:   -isystem %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/no-change-same-session.c --check-prefix=CHECK-NO-VALIDATION-OR-BUILD --allow-empty
// RUN: %clang @%t/ctx.rsp %t/no-change-same-session.c -DCTX=2 \
// RUN:   -I %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/no-change-same-session.c  --check-prefix=CHECK-VALIDATION-ONLY
// RUN: %clang @%t/ctx.rsp %t/no-change-same-session.c -DCTX=3 \
// RUN:   -I %t/include -fmodules-validate-system-headers -Xclang -fno-modules-force-validate-user-headers \
// RUN:     2>&1 | FileCheck %t/no-change-same-session.c --check-prefix=CHECK-NO-VALIDATION-OR-BUILD --allow-empty
#include "foo.h"
// CHECK-NO-VALIDATION-OR-BUILD-NOT: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-NO-VALIDATION-OR-BUILD-NOT: building module 'Foo'
// CHECK-VALIDATION-ONLY: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-VALIDATION-ONLY-NOT: building module 'Foo'

//--- change-same-session.c
// Populated module cache in the same build session with changed inputs.
// Validation only happens when it's forced for user headers and results in compilation.
// RUN: sleep 1
// RUN: touch %t/include/foo.h
// RUN: sleep 1
// RUN: %clang @%t/ctx.rsp %t/change-same-session.c -DCTX=1 \
// RUN:   -isystem %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/change-same-session.c --check-prefix=CHECK-NO-VALIDATION-OR-BUILD --allow-empty
// RUN: %clang @%t/ctx.rsp %t/change-same-session.c -DCTX=2 \
// RUN:   -I %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/change-same-session.c --check-prefix=CHECK-VALIDATION-AND-BUILD
// RUN: %clang @%t/ctx.rsp %t/change-same-session.c -DCTX=3 \
// RUN:   -I %t/include -fmodules-validate-system-headers -Xclang -fno-modules-force-validate-user-headers \
// RUN:     2>&1 | FileCheck %t/change-same-session.c --check-prefix=CHECK-NO-VALIDATION-OR-BUILD --allow-empty
#include "foo.h"
// CHECK-NO-VALIDATION-OR-BUILD-NOT: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-NO-VALIDATION-OR-BUILD-NOT: building module 'Foo'
// CHECK-VALIDATION-AND-BUILD: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-VALIDATION-AND-BUILD: building module 'Foo'

//--- change-new-session.c
// Populated module cache in a new build session with changed inputs.
// All configurations validate and recompile.
// RUN: sleep 1
// RUN: touch %t/include/foo.h
// RUN: sleep 1
// RUN: touch %t/module-cache/session.timestamp
// RUN: sleep 1
// RUN: %clang @%t/ctx.rsp %t/change-new-session.c -DCTX=1 \
// RUN:   -isystem %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/change-new-session.c --check-prefixes=CHECK,CHECK-VALIDATE-ONCE
// NOTE: Forced user headers validation causes redundant validation of the just-built module.
// RUN: %clang @%t/ctx.rsp %t/change-new-session.c -DCTX=2 \
// RUN:   -I %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/change-new-session.c --check-prefixes=CHECK,CHECK-FORCE-VALIDATE-TWICE
// RUN: %clang @%t/ctx.rsp %t/change-new-session.c -DCTX=3 \
// RUN:   -I %t/include -fmodules-validate-system-headers -Xclang -fno-modules-force-validate-user-headers \
// RUN:     2>&1 | FileCheck %t/change-new-session.c --check-prefixes=CHECK,CHECK-VALIDATE-ONCE
#include "foo.h"
// CHECK: validating {{[0-9]+}} input files in module 'Foo'
// CHECK: building module 'Foo'
// CHECK-VALIDATE-ONCE-NOT: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-FORCE-VALIDATE-TWICE: validating {{[0-9]+}} input files in module 'Foo'

//--- no-change-new-session-twice.c
// Populated module cache in a new build session with unchanged inputs.
// At first, all configurations validate but don't recompile.
// RUN: sleep 1
// RUN: touch %t/module-cache/session.timestamp
// RUN: sleep 1
// RUN: %clang @%t/ctx.rsp %t/no-change-new-session-twice.c -DCTX=1 \
// RUN:   -isystem %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/no-change-new-session-twice.c --check-prefix=CHECK-ONCE
// RUN: %clang @%t/ctx.rsp %t/no-change-new-session-twice.c -DCTX=2 \
// RUN:   -I %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/no-change-new-session-twice.c --check-prefix=CHECK-ONCE
// RUN: %clang @%t/ctx.rsp %t/no-change-new-session-twice.c -DCTX=3 \
// RUN:   -I %t/include -fmodules-validate-system-headers -Xclang -fno-modules-force-validate-user-headers \
// RUN:     2>&1 | FileCheck %t/no-change-new-session-twice.c --check-prefix=CHECK-ONCE
//
// Then, only the forced user header validation performs redundant validation (but no compilation).
// All other configurations do not validate and do not compile.
// RUN: sleep 1
// RUN: %clang @%t/ctx.rsp %t/no-change-new-session-twice.c -DCTX=1 \
// RUN:   -isystem %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/no-change-new-session-twice.c --check-prefix=CHECK-NOT-TWICE --allow-empty
// NOTE: Forced user headers validation causes redundant validation of the just-validated module.
// RUN: %clang @%t/ctx.rsp %t/no-change-new-session-twice.c -DCTX=2 \
// RUN:   -I %t/include -fmodules-validate-system-headers \
// RUN:     2>&1 | FileCheck %t/no-change-new-session-twice.c --check-prefix=CHECK-ONCE
// RUN: %clang @%t/ctx.rsp %t/no-change-new-session-twice.c -DCTX=3 \
// RUN:   -I %t/include -fmodules-validate-system-headers -Xclang -fno-modules-force-validate-user-headers \
// RUN:     2>&1 | FileCheck %t/no-change-new-session-twice.c --check-prefix=CHECK-NOT-TWICE --allow-empty
#include "foo.h"
// CHECK-ONCE: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-ONCE-NOT: building module 'Foo'
// CHECK-NOT-TWICE-NOT: validating {{[0-9]+}} input files in module 'Foo'
// CHECK-NOT-TWICE-NOT: building module 'Foo'
