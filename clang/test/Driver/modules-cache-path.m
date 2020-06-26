// RUN: env USERNAME=asdf LOGNAME=asdf %clang -fmodules -### %s 2>&1 | FileCheck %s -check-prefix=CHECK-SET
// CHECK-SET: -fmodules-cache-path={{.*}}org.llvm.clang.asdf{{[/\\]+}}ModuleCache

// RUN: %clang -fmodules -### %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: -fmodules-cache-path={{.*}}org.llvm.clang.{{[A-Za-z0-9_]*[/\\]+}}ModuleCache
