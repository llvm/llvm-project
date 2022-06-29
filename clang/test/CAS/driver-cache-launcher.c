// REQUIRES: shell

// RUN: rm -rf %t && mkdir %t
// RUN: echo "#!/bin/sh"              > %t/clang
// RUN: echo "echo run some compiler with opts \$*" >> %t/clang
// RUN: chmod +x %t/clang
// RUN: ln -s %clang %t/clang-symlink-outside-bindir

// 'clang-cache' launcher invokes itself, enables caching.
// RUN: env CLANG_CACHE_CAS_PATH=%t/cas clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG -DPREFIX=%t
// RUN: env CLANG_CACHE_CAS_PATH=%t/cas clang-cache %clang++ -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANGPP -DPREFIX=%t
// RUN: env CLANG_CACHE_CAS_PATH=%t/cas clang-cache %t/clang-symlink-outside-bindir -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG -DPREFIX=%t
// RUN: env CLANG_CACHE_CAS_PATH=%t/cas PATH="%t:$PATH" clang-cache clang-symlink-outside-bindir -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG -DPREFIX=%t

// CLANG: "-cc1depscan" "-fdepscan=auto"
// CLANG: "-fcas-path" "[[PREFIX]]/cas"
// CLANG: "-greproducible"
// CLANG: "-x" "c"

// CLANGPP: "-cc1depscan" "-fdepscan=auto"
// CLANGPP: "-fcas-path" "[[PREFIX]]/cas"
// CLANGPP: "-greproducible"
// CLANGPP: "-x" "c++"

// RUN: env CLANG_CACHE_CAS_PATH=%t/cas cache-build-session clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=SESSION -DPREFIX=%t
// SESSION: "-cc1depscan" "-fdepscan=daemon" "-fdepscan-share-identifier"
// SESSION: "-fcas-path" "[[PREFIX]]/cas"
// SESSION: "-greproducible"

// RUN: cp -R %S/Inputs/cmake-build %t/cmake-build
// RUN: pushd %t/cmake-build
// RUN: cache-build-session -prefix-map-cmake -v echo 2>&1 | FileCheck %s -check-prefix=SESSION-CMAKE-PREFIX
// RUN: env CLANG_CACHE_CAS_PATH=%t/cas cache-build-session -prefix-map-cmake clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG-CMAKE-PREFIX -DPREFIX=%t
// RUN: popd

// SESSION-CMAKE-PREFIX: note: setting LLVM_CACHE_PREFIX_MAPS=/llvm/build=/^build;/llvm/llvm-project/llvm=/^src;/llvm/llvm-project/clang=/^src-clang;/llvm/llvm-project/clang-tools-extra=/^src-clang-tools-extra;/llvm/llvm-project/third-party/benchmark=/^src-benchmark;/llvm/llvm-project/other/benchmark=/^src-benchmark-1;/llvm/llvm-project/another/benchmark=/^src-benchmark-2{{$}}
// SESSION-CMAKE-PREFIX: note: setting LLVM_CACHE_BUILD_SESSION_ID=

// CLANG-CMAKE-PREFIX: "-cc1depscan" "-fdepscan=daemon" "-fdepscan-share-identifier"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map-sdk=/^sdk" "-fdepscan-prefix-map-toolchain=/^toolchain"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/build=/^build"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/llvm-project/llvm=/^src"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/llvm-project/clang=/^src-clang"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/llvm-project/clang-tools-extra=/^src-clang-tools-extra"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/llvm-project/third-party/benchmark=/^src-benchmark"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/llvm-project/other/benchmark=/^src-benchmark-1"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map=/llvm/llvm-project/another/benchmark=/^src-benchmark-2"

// Make sure `cache-build-session` can invoke an executable script.
// RUN: cache-build-session %t/clang -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=SESSION-SCRIPT -DSRC=%s -DPREFIX=%t
// SESSION-SCRIPT: run some compiler with opts -c [[SRC]] -o [[PREFIX]].o

// 'clang-cache' launcher invokes a different clang, does normal non-caching launch.
// RUN: env CLANG_CACHE_CAS_PATH=%t/cas clang-cache %t/clang -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=OTHERCLANG -DSRC=%s -DPREFIX=%t
// OTHERCLANG: warning: clang-cache invokes a different clang binary than itself, it will perform a normal non-caching invocation of the compiler
// OTHERCLANG-NEXT: run some compiler with opts -c [[SRC]] -o [[PREFIX]].o

// RUN: not clang-cache %t/nonexistent -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=NONEXISTENT
// NONEXISTENT: error: clang-cache failed to execute compiler

// RUN: not clang-cache 2>&1 | FileCheck %s -check-prefix=NOCOMMAND
// NOCOMMAND: error: missing compiler command for clang-cache
