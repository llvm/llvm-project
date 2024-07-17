// Check that implicit modules builds give correct diagnostics, even when
// reusing a module built with strong -Werror flags.
//
// Clear the caches.
// RUN: rm -rf %t.cache %t-pragma.cache
//
// Build with -Werror, then with -W, and then with neither.
// RUN: not %clang_cc1 -triple x86_64-apple-darwin16 -fsyntax-only -fmodules \
// RUN:   -Werror=shorten-64-to-32 \
// RUN:   -I%S/Inputs/implicit-built-Werror-using-W -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.cache -x c++ %s 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-ERROR
// RUN: %clang_cc1 -triple x86_64-apple-darwin16 -fsyntax-only -fmodules \
// RUN:   -Wshorten-64-to-32 \
// RUN:   -I%S/Inputs/implicit-built-Werror-using-W -fimplicit-module-maps \
// RUN:   -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t.cache -x c++ %s 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-WARN
// RUN: %clang_cc1 -triple x86_64-apple-darwin16 -fsyntax-only -fmodules \
// RUN:   -I%S/Inputs/implicit-built-Werror-using-W -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.cache -x c++ %s 2>&1 \
// RUN: | FileCheck %s -allow-empty
//
// In the presence of a warning pragma, build with -Werror and then without.
// RUN: %clang_cc1 -triple x86_64-apple-darwin16 -fsyntax-only -fmodules \
// RUN:   -DUSE_PRAGMA=1 -Werror=shorten-64-to-32 \
// RUN:   -I%S/Inputs/implicit-built-Werror-using-W -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t-pragma.cache -x c++ %s 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-WARN
// RUN: %clang_cc1 -triple x86_64-apple-darwin16 -fsyntax-only -fmodules \
// RUN:   -DUSE_PRAGMA=1 \
// RUN:   -I%S/Inputs/implicit-built-Werror-using-W -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t-pragma.cache -x c++ %s 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-WARN

// Test an error pragma.
// RUN: not %clang_cc1 -triple x86_64-apple-darwin16 -fsyntax-only -fmodules \
// RUN:   -DUSE_PRAGMA=2 -Wshorten-64-to-32 \
// RUN:   -I%S/Inputs/implicit-built-Werror-using-W -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t-pragma.cache -x c++ %s 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-ERROR
#include <convert.h>

long long foo() { return convert<long long>(0); }

// CHECK-ERROR: error: implicit conversion
// CHECK-WARN: warning: implicit conversion
// CHECK-NOT: error: implicit conversion
// CHECK-NOT: warning: implicit conversion
