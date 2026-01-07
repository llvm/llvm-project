// REQUIRES: shell

// RUN: rm -rf %t && mkdir %t
// RUN: echo "#!/bin/sh"              > %t/clang
// RUN: echo "echo run some compiler with opts \$*" >> %t/clang
// RUN: chmod +x %t/clang
// RUN: ln -s %clang %t/clang-symlink-outside-bindir

// 'clang-cache' launcher invokes itself, enables caching.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG -DPREFIX=%t
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang++ -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANGPP -DPREFIX=%t
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %t/clang-symlink-outside-bindir -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG -DPREFIX=%t
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas PATH="%t:$PATH" %clang-cache clang-symlink-outside-bindir -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=CLANG -DPREFIX=%t

// CLANG: "-cc1depscan" "-fdepscan=auto"
// CLANG: "-fcas-path" "[[PREFIX]]/cas"
// CLANG: "-greproducible"
// CLANG: "-x" "c"

// CLANGPP: "-cc1depscan" "-fdepscan=auto"
// CLANGPP: "-fcas-path" "[[PREFIX]]/cas"
// CLANGPP: "-greproducible"
// CLANGPP: "-x" "c++"

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=INCLUDE-TREE -DPREFIX=%t
// INCLUDE-TREE: "-cc1depscan" "-fdepscan=auto"

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas cache-build-session %clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=SESSION -DPREFIX=%t
// SESSION: "-cc1depscan" "-fdepscan=daemon" "-fdepscan-share-identifier"
// SESSION: "-fcas-path" "[[PREFIX]]/cas"
// SESSION: "-greproducible"

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_SCAN_DAEMON_SOCKET_PATH=%t/scand cache-build-session %clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=SPECIFIC-DAEMON -DPREFIX=%t
// SPECIFIC-DAEMON-NOT: "-fdepscan-share-identifier"
// SPECIFIC-DAEMON: "-cc1depscan" "-fdepscan=daemon" "-fdepscan-daemon=[[PREFIX]]/scand"

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CACHE_PLUGIN_PATH=%t/plugin LLVM_CACHE_PLUGIN_OPTIONS=some-opt=value:opt2=val2 \
// RUN:   %clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=PLUGIN -DPREFIX=%t
// PLUGIN: "-fcas-path" "[[PREFIX]]/cas"
// PLUGIN: "-fcas-plugin-path" "[[PREFIX]]/plugin"
// PLUGIN: "-fcas-plugin-option" "some-opt=value"
// PLUGIN: "-fcas-plugin-option" "opt2=val2"

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN: %clang-cache %clang -target x86_64-apple-darwin10 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=ENABLE-MCCAS -DPREFIX=%t
// ENABLE-MCCAS: "-fcas-path" "[[PREFIX]]/cas"
// ENABLE-MCCAS: "-fcas-backend"
// ENABLE-MCCAS: "-mllvm"
// ENABLE-MCCAS: "-cas-friendly-debug-info"

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_VERIFY_MCCAS=1 \
// RUN: %clang-cache %clang -target x86_64-apple-darwin10 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=ENABLE-MCCAS-VERIFY -DPREFIX=%t
// ENABLE-MCCAS-VERIFY: "-fcas-path" "[[PREFIX]]/cas"
// ENABLE-MCCAS-VERIFY: "-fcas-backend"
// ENABLE-MCCAS-VERIFY: "-fcas-backend-mode=verify"
// ENABLE-MCCAS-VERIFY: "-mllvm"
// ENABLE-MCCAS-VERIFY: "-cas-friendly-debug-info"

// ELF/COFF platforms do not support mccas.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN: %clang-cache %clang -target x86_64-unknown-linux-gnu -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=DISABLE-MCCAS -DPREFIX=%t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN: %clang-cache %clang -target x86_64-unknown-windows-msvc -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=DISABLE-MCCAS -DPREFIX=%t

// Forcing mccas gives an error on ELF/COFF.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN: not %clang -target x86_64-unknown-linux-gnu -Xclang -fcas-backend -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=MCCAS_NOT_ALLOWED
// MCCAS_NOT_ALLOWED: error: -fcas-backend is incompatible with target 'x86_64-unknown-linux-gnu'; requires mach-o object format

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_DISABLE_MCCAS=1 \
// RUN: %clang-cache %clang -target x86_64-apple-darwin10 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=DISABLE-MCCAS -DPREFIX=%t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH=%t/ccremote \
// RUN: %clang-cache %clang -target x86_64-apple-darwin10 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=DISABLE-MCCAS -DPREFIX=%t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH=%t/ccremote CLANG_CACHE_DISABLE_MCCAS=1 \
// RUN: %clang-cache %clang -target x86_64-apple-darwin10 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=DISABLE-MCCAS -DPREFIX=%t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH=%t/ccremote CLANG_CACHE_DISABLE_MCCAS=1 CLANG_CACHE_VERIFY_MCCAS=1 \
// RUN: %clang-cache %clang -target x86_64-apple-darwin10 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=DISABLE-MCCAS -DPREFIX=%t

// DISABLE-MCCAS-NOT: "-fcas-backend"
// DISABLE-MCCAS-NOT: "-fcas-backend-mode=verify"
// DISABLE-MCCAS-NOT: "-mllvm" "-cas-friendly-debug-info"



// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH=%t/ccremote %clang-cache %clang -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=REMOTE -DPREFIX=%t
// REMOTE: "-fcompilation-caching-service-path" "[[PREFIX]]/ccremote"

// Using multi-arch invocation.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -target x86_64-apple-macos12 -arch x86_64 -arch arm64 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=MULTIARCH
// MULTIARCH: "-cc1depscan" "-fdepscan=auto"
// MULTIARCH: "-triple" "x86_64-apple-macosx12.0.0"
// MULTIARCH: "-cc1depscan" "-fdepscan=auto"
// MULTIARCH: "-triple" "arm64-apple-macosx12.0.0"

// RUN: cp -R %S/Inputs/cmake-build %t/cmake-build
// RUN: pushd %t/cmake-build
// RUN: cache-build-session -prefix-map-cmake -v echo 2>&1 | FileCheck %s -check-prefix=SESSION-CMAKE-PREFIX
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas cache-build-session -prefix-map-cmake %clang-cache %clang -c %s -o %t.o -isysroot %S/Inputs/SDK -resource-dir %S/Inputs/toolchain_dir/lib/clang/1000 -### 2>&1 | FileCheck %s -check-prefix=CLANG-CMAKE-PREFIX -DPREFIX=%t -DINPUTS=%S/Inputs
// RUN: popd

// SESSION-CMAKE-PREFIX: note: setting LLVM_CACHE_PREFIX_MAPS=/llvm/build=/^build;/llvm/llvm-project/llvm=/^src;/llvm/llvm-project/clang=/^src-clang;/llvm/llvm-project/clang-tools-extra=/^src-clang-tools-extra;/llvm/llvm-project/third-party/benchmark=/^src-benchmark;/llvm/llvm-project/other/benchmark=/^src-benchmark-1;/llvm/llvm-project/another/benchmark=/^src-benchmark-2{{$}}
// SESSION-CMAKE-PREFIX: note: setting LLVM_CACHE_BUILD_SESSION_ID=

// CLANG-CMAKE-PREFIX: "-cc1depscan" "-fdepscan=daemon" "-fdepscan-share-identifier"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "[[INPUTS]]/SDK" "/^sdk"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "[[INPUTS]]/toolchain_dir" "/^toolchain"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/build" "/^build"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/llvm-project/llvm" "/^src"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/llvm-project/clang" "/^src-clang"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/llvm-project/clang-tools-extra" "/^src-clang-tools-extra"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/llvm-project/third-party/benchmark" "/^src-benchmark"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/llvm-project/other/benchmark" "/^src-benchmark-1"
// CLANG-CMAKE-PREFIX: "-fdepscan-prefix-map" "/llvm/llvm-project/another/benchmark" "/^src-benchmark-2"

// Make sure `cache-build-session` can invoke an executable script.
// RUN: cache-build-session %t/clang -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=SESSION-SCRIPT -DSRC=%s -DPREFIX=%t
// SESSION-SCRIPT: run some compiler with opts -c [[SRC]] -o [[PREFIX]].o

// 'clang-cache' launcher invokes a different clang, does normal non-caching launch.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %t/clang -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=OTHERCLANG -DSRC=%s -DPREFIX=%t
// OTHERCLANG: warning: caching disabled because clang-cache invokes a different clang binary than itself
// OTHERCLANG-NEXT: run some compiler with opts -c [[SRC]] -o [[PREFIX]].o

// RUN: %clang-cache %clang -x objective-c++ -fmodules -fmodules-cache-path=%t/mcp -fno-cxx-modules -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=NONMOD -DPREFIX=%t
// RUN: %clang-cache %clang -x c++ -fmodules -fmodules-cache-path=%t/mcp -fno-cxx-modules -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=NONMOD -DPREFIX=%t
// RUN: %clang-cache %clang -x objective-c -fmodules -fmodules-cache-path=%t/mcp -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=MOD -DPREFIX=%t
// RUN: %clang-cache %clang -x objective-c++ -fmodules -fmodules-cache-path=%t/mcp -fcxx-modules -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=MOD -DPREFIX=%t
// RUN: %clang-cache %clang -x c++ -fmodules -fmodules-cache-path=%t/mcp -fcxx-modules -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=MOD -DPREFIX=%t
// NONMOD: "-cc1depscan"
// MOD: warning: caching disabled because -fmodules is enabled
// MOD-NOT: "-cc1depscan"

// RUN: touch %t/t.s
// RUN: %clang-cache %clang -target arm64-apple-macosx12 -c %t/t.s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=ASM
// ASM: warning: caching disabled because assembler language mode is enabled
// ASM-NOT: "-cc1depscan"

// RUN: env AS_SECURE_LOG_FILE=%t/log %clang-cache %clang -target arm64-apple-macosx12 -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=AS_SECURE_LOG_FILE
// AS_SECURE_LOG_FILE: warning: caching disabled because AS_SECURE_LOG_FILE is set
// AS_SECURE_LOG_FILE-NOT: "-cc1depscan"

// RUN: env LLVM_CACHE_WARNINGS=-Wno-clang-cache %clang-cache %clang -x c++ -fmodules -fmodules-cache-path=%t/mcp -fcxx-modules -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=MOD_HIDE -DPREFIX=%t
// MOD_HIDE-NOT: warning: caching disabled
// MOD_HIDE-NOT: "-cc1depscan"

// RUN: env LLVM_CACHE_WARNINGS=-Werror=clang-cache not %clang-cache %clang -x c++ -fmodules -fmodules-cache-path=%t/mcp -fcxx-modules -c %s -o %t.o -### 2>&1 | FileCheck %s -check-prefix=MOD_ERR -DPREFIX=%t
// MOD_ERR: error: caching disabled because -fmodules is enabled

// RUN: not %clang-cache %t/nonexistent -c %s -o %t.o 2>&1 | FileCheck %s -check-prefix=NONEXISTENT
// NONEXISTENT: error: clang-cache failed to execute compiler

// RUN: not %clang-cache 2>&1 | FileCheck %s -check-prefix=NOCOMMAND
// NOCOMMAND: error: missing compiler command for clang-cache

// Link-only job should not attempt to cache.
// RUN: touch %t.o
// RUN: %clang-cache %clang -target arm64-apple-macosx12 %t.o -o %t -Wl,-ObjC -### 2>&1 | FileCheck %s -check-prefix=STATIC_LINK
// STATIC_LINK-NOT: warning:
// STATIC_LINK-NOT: "-cc1depscan"
// STATIC_LINK: "{{[^"]*}}ld"

// Unused option warning should only be emitted once.
// RUN: touch %t.o
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -target arm64-apple-macosx12 -fsyntax-only %s -Wl,-ObjC 2>&1 | FileCheck %s -check-prefix=UNUSED_OPT
// UNUSED_OPT-NOT: warning:
// UNUSED_OPT: warning: -Wl,-ObjC: 'linker' input unused
// UNUSED_OPT-NOT: warning:
