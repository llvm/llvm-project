// Run lines are sensitive to line numbers and come below the code.

#include "retain-comments-from-system-headers.h"

/**
 * user_function
 * \param a Aaa.
 */
int user_function(int a);

// RUN: rm -rf %t/cache
// RUN: c-index-test -test-load-source all %s -I %S/Inputs | FileCheck %s
// RUN: c-index-test -test-load-source all %s -fretain-comments-from-system-headers -I %S/Inputs | FileCheck %s -check-prefix=CHECK-RETAIN

// Modules:
// RUN: c-index-test -test-load-source all %s -I %S/Inputs -fmodules -fmodules-cache-path=%t/cache -fmodule-map-file=%S/Inputs/retain-comments-from-system-headers-module.map | FileCheck %s
// RUN: c-index-test -test-load-source all %s -fretain-comments-from-system-headers -I %S/Inputs -fmodules -fmodules-cache-path=%t/cache -fmodule-map-file=%S/Inputs/retain-comments-from-system-headers-module.map | FileCheck %s -check-prefix=CHECK-RETAIN

// CHECK: retain-comments-from-system-headers.h:7:5: FunctionDecl=system_function:7:5 Extent=[7:1 - 7:27]
// CHECK: retain-comments-from-system-headers.c:9:5: FunctionDecl=user_function:9:5 RawComment=[/**\n * user_function\n * \param a Aaa.\n */] RawCommentRange=[5:1 - 8:4] BriefComment=[user_function]

// CHECK-RETAIN: retain-comments-from-system-headers.h:7:5: FunctionDecl=system_function:7:5 RawComment=[/**\n * system_function\n * \param a Aaa.\n */] RawCommentRange=[3:1 - 6:4] BriefComment=[system_function]
// CHECK-RETAIN: retain-comments-from-system-headers.c:9:5: FunctionDecl=user_function:9:5 RawComment=[/**\n * user_function\n * \param a Aaa.\n */] RawCommentRange=[5:1 - 8:4] BriefComment=[user_function]
