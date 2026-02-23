// RUN: rm -rf %t && mkdir -p %t

// REQUIRES: system-darwin

// Check if -fwrite-output-hash-xattr on a cache miss with file based caching
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_DISABLE_MCCAS=1 %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fwrite-output-hash-xattr -g -c %s -o %t/test.o -Rcompile-job-cache 2> %t/diag_miss
// RUN: xattr -lx %t/test.o | FileCheck %s
// RUN: FileCheck %s -check-prefix=MISS -input-file %t/diag_miss
// RUN: rm -f %t/test.o

// Check if -fwrite-output-hash-xattr works on a cache hit with file based caching
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_DISABLE_MCCAS=1 %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fwrite-output-hash-xattr -g -c %s -o %t/test.o -Rcompile-job-cache 2> %t/diag_hit
// RUN: xattr -lx %t/test.o | FileCheck %s
// RUN: rm -f %t/test.o
// RUN: FileCheck %s -check-prefix=HIT -input-file %t/diag_hit

// The following checks the hash schema name and the 32-byte hash size.
// CHECK: com.apple.clang.cas_output_hash:
// CHECK: 00000000  6C 6C 76 6D 2E 63 61 73 2E 62 75 69 6C 74 69 6E  |llvm.cas.builtin|
// CHECK: 00000010  2E 76 32 5B 42 4C 41 4B 45 33 5D 00 20 00 00 00  |.v2[BLAKE3]. ...|
// CHECK: 00000020  {{(([A-F0-9]{2} ){16})}}|
// CHECK: 00000030  {{(([A-F0-9]{2} ){16})}}|

// HIT: compile job cache hit for
// MISS: compile job cache miss for

// Check errors.
// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fwrite-output-hash-xattr -g -c %s -o %t/test.o 2>&1 | FileCheck %s -check-prefix=INCOMPAT_MCCAS
// INCOMPAT_MCCAS: error: '-fcas-backend' is incompatible with '-fwrite-output-hash-xattr'

// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_DISABLE_MCCAS=1 %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fwrite-output-hash-xattr -Xclang -fcasid-output -g -c %s -o %t/test.o 2>&1 | FileCheck %s -check-prefix=INCOMPAT_CASIDOUT
// INCOMPAT_CASIDOUT: error: '-fcasid-output' is incompatible with '-fwrite-output-hash-xattr'

// Check libclang replay.
// RUN: cat %t/diag_hit | grep llvmcas | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key

// RUN: c-index-test core -replay-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:   -working-dir %t \
// RUN: -- -target x86_64-apple-macos11 -fwrite-output-hash-xattr -emit-obj %s -o %t/test.o
// RUN: xattr -lx %t/test.o | FileCheck %s
