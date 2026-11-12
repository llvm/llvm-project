// REQUIRES: case-insensitive-filesystem

// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fapinotes-modules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/Frameworks -I %S/Inputs/Headers %s 2>&1 | FileCheck %s

// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fapinotes-modules -fimplicit-module-maps -fmodules-cache-path=%t -iframework %S/Inputs/Frameworks -isystem %S/Inputs/Headers %s -Werror

// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fapinotes-modules -fimplicit-module-maps -fmodules-cache-path=%t -iframework %S/Inputs/Frameworks -isystem %S/Inputs/Headers %s -Wnonportable-private-system-apinotes-path 2>&1 | FileCheck %s

#include <ModuleWithWrongCase.h>
#include <ModuleWithWrongCasePrivate.h>
#include <FrameworkWithWrongCase/FrameworkWithWrongCase.h>
#include <FrameworkWithWrongCasePrivate/FrameworkWithWrongCasePrivate.h>
#include <FrameworkWithActualPrivateModule/FrameworkWithActualPrivateModule_Private.h>

// CHECK-NOT: warning:
// CHECK: warning: private API notes file for module 'ModuleWithWrongCasePrivate' should be named 'ModuleWithWrongCasePrivate_private.apinotes', not 'ModuleWithWrongCasePrivate_Private.apinotes'
// CHECK-NOT: warning:
// CHECK: warning: private API notes file for module 'FrameworkWithWrongCasePrivate' should be named 'FrameworkWithWrongCasePrivate_private.apinotes', not 'FrameworkWithWrongCasePrivate_Private.apinotes'
// CHECK-NOT: warning:
