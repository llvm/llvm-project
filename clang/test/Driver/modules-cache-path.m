// RUN: env -u CLANG_MODULE_CACHE_PATH %clang -fmodules -### %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: -fmodules-cache-path={{.*}}clang{{[/\\]+}}ModuleCache

// RUN: env CLANG_MODULE_CACHE_PATH=/dev/null \
// RUN:   %clang -fmodules -### %s 2>&1 | FileCheck %s -check-prefix=OVERRIDE
// OVERRIDE: -fmodules-cache-path=/dev/null

// RUN: env CLANG_MODULE_CACHE_PATH= \
// RUN:   %clang -fmodules -### %s 2>&1 | FileCheck %s -check-prefix=DISABLE
// DISABLE-NOT: -fmodules-cache-path=
