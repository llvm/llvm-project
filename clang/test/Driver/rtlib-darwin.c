// RUN: %clang -target x86_64-apple-darwin -resource-dir=%S/Inputs/resource_dir --rtlib=compiler-rt -### %s 2>&1 | FileCheck %s -check-prefix DARWIN-COMPILER-RT
// RUN: %clang -target x86_64-apple-darwin -resource-dir=%S/Inputs/resource_dir --rtlib=platform -### %s 2>&1 | FileCheck %s -check-prefix DARWIN-COMPILER-RT
// RUN: not %clang %s -target x86_64-apple-darwin --rtlib=libgcc 2>&1 | FileCheck %s -check-prefix CHECK-ERROR

// DARWIN-COMPILER-RT: "{{.*}}clang_rt.osx{{.*}}"
// CHECK-ERROR: unsupported runtime library 'libgcc' for platform 'darwin'
