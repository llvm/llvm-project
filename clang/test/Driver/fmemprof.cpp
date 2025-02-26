// RUN: %clangxx --target=x86_64-linux-gnu -fmemory-profile %s -### 2>&1 | FileCheck %s
// RUN: %clangxx --target=x86_64-linux-gnu -fmemory-profile=foo %s -### 2>&1 | FileCheck %s --check-prefix=DIR
// RUN: %clangxx --target=x86_64-linux-gnu -fmemory-profile -fno-memory-profile %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clangxx --target=x86_64-linux-gnu -fmemory-profile=foo -fno-memory-profile %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// CHECK: "-cc1" {{.*}} "-fmemory-profile"
// CHECK: ld{{.*}}libclang_rt.memprof{{.*}}libclang_rt.memprof_cxx
// DIR: "-cc1" {{.*}} "-fmemory-profile=foo"
// DIR: ld{{.*}}libclang_rt.memprof{{.*}}libclang_rt.memprof_cxx
// OFF-NOT: "-fmemory-profile"
// OFF-NOT: libclang_rt.memprof

// RUN: %clangxx --target=x86_64-linux-gnu -fmemory-profile-use=foo %s -### 2>&1 | FileCheck %s --check-prefix=USE
// USE: "-cc1" {{.*}} "-fmemory-profile-use=foo"

// RUN: not %clangxx --target=x86_64-linux-gnu -fmemory-profile -fmemory-profile-use=foo %s -### 2>&1 | FileCheck %s --check-prefix=CONFLICTWITHMEMPROFINSTR
// CONFLICTWITHMEMPROFINSTR: error: invalid argument '-fmemory-profile-use=foo' not allowed with '-fmemory-profile'

// RUN: not %clangxx --target=x86_64-linux-gnu -fprofile-generate -fmemory-profile-use=foo %s -### 2>&1 | FileCheck %s --check-prefix=CONFLICTWITHPGOINSTR
// CONFLICTWITHPGOINSTR: error: invalid argument '-fmemory-profile-use=foo' not allowed with '-fprofile-generate'

// Test that we export the __memprof_default_options_str on Darwin because it has WeakAnyLinkage
// RUN: %clangxx --target=arm64-apple-ios -fmemory-profile %s -### 2>&1 | FileCheck %s -check-prefix=EXPORT-BASE --implicit-check-not=exported_symbol
// RUN: %clangxx --target=x86_64-linux-gnu -fmemory-profile %s -### 2>&1 | FileCheck %s -check-prefix=EXPORT-BASE --implicit-check-not=exported_symbol
// RUN: %clangxx --target=arm64-apple-ios -fmemory-profile -exported_symbols_list /dev/null %s -### 2>&1 | FileCheck %s --check-prefixes=EXPORT-BASE,EXPORT
// FIXME: Darwin needs to link in the runtime, then we can use the regular CHECK prefix
// EXPORT-BASE: "-cc1" {{.*}} "-fmemory-profile"
// EXPORT: "-exported_symbol" "___memprof_default_options_str"
