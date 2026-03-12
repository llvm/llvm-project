// RUN: rm -rf %t && mkdir %t
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas cache-build-session -v %clang-cache %clang_cl -### -c -- %s 2>&1 | FileCheck %s

// CHECK: note: setting LLVM_CACHE_BUILD_SESSION_ID=[[SESSION_ID:[0-9-]+]]
// CHECK: "-fdepscan-share-identifier" "[[SESSION_ID]]"
