// Test that DW_AT_comp_dir is preserved when needed with compilation caching.
// This option reduces cache hits if working directory changes, so we only
// preserve it if it could be required to resolve a relative path.

// REQUIRES: ondisk_cas
// REQUIRES: x86-registered-target

// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

//--- src/test.c
#include "test.h"
void foo(void) {}

//--- header_dir/test.h
void bar(void) {}

// Absolute paths only. DW_AT_comp_dir should not be present.

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I %t/header_dir -o %t/abs-cache.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/abs-cache.o \
// RUN:   | FileCheck %s --check-prefix=ABS-CACHE

// MISS: remark: compile job cache miss
// HIT:  remark: compile job cache hit

// ABS-CACHE:     DW_TAG_compile_unit
// ABS-CACHE-NOT: DW_AT_comp_dir

// Relative include path. DW_AT_comp_dir should be preserved.

// Cache miss. Run from %t so that the relative -I path resolves.
// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I header_dir -o %t/rel-cache.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/rel-cache.o \
// RUN:   | FileCheck %s --check-prefix=REL-CACHE

// REL-CACHE: DW_TAG_compile_unit
// REL-CACHE: DW_AT_comp_dir

// Cache hit.
// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I header_dir -o %t/rel-cache-hit.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=HIT
// RUN: llvm-dwarfdump --debug-info %t/rel-cache-hit.o \
// RUN:   | FileCheck %s --check-prefix=REL-CACHE

// Relative include path + -working-directory, which makes paths absolute, so
// DW_AT_comp_dir should be cleared.

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I header_dir -working-directory %t \
// RUN:   -o %t/wd-cache-hit.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/wd-cache-hit.o \
// RUN:   | FileCheck %s --check-prefix=WD-CACHE

// WD-CACHE:     DW_TAG_compile_unit
// WD-CACHE-NOT: DW_AT_comp_dir

// Prefix mapping with relative paths.
// DW_AT_comp_dir should be the prefix-mapped path.

// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:     LLVM_CACHE_PREFIX_MAPS="%t=/^build" \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I header_dir -o %t/prefix-cache.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/prefix-cache.o \
// RUN:   | FileCheck %s --check-prefix=PREFIX-CACHE

// PREFIX-CACHE:     DW_TAG_compile_unit
// PREFIX-CACHE:     DW_AT_comp_dir ("/^build")

// Relative input file path. DW_AT_comp_dir should be preserved.

// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c src/test.c -I %t/header_dir -o %t/relinput.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/relinput.o \
// RUN:   | FileCheck %s --check-prefix=REL-CACHE

// Relative sysroot. DW_AT_comp_dir should be preserved.

// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I %t/header_dir -isysroot sysroot \
// RUN:   -o %t/relsysroot.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/relsysroot.o \
// RUN:   | FileCheck %s --check-prefix=REL-CACHE

// Relative resource dir. DW_AT_comp_dir should be preserved.

// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I %t/header_dir \
// RUN:   -Xclang -resource-dir -Xclang resource \
// RUN:   -o %t/relresdir.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/relresdir.o \
// RUN:   | FileCheck %s --check-prefix=REL-CACHE

// Relative system header prefix. DW_AT_comp_dir should be preserved.

// RUN: cd %t && env LLVM_CACHE_CAS_PATH=%t/cas \
// RUN:   %clang-cache %clang -target x86_64-apple-macos11 -g \
// RUN:   -c %t/src/test.c -I %t/header_dir \
// RUN:   -Xclang --system-header-prefix=header_dir/ \
// RUN:   -o %t/relsyshdrpfx.o \
// RUN:   -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=MISS
// RUN: llvm-dwarfdump --debug-info %t/relsyshdrpfx.o \
// RUN:   | FileCheck %s --check-prefix=REL-CACHE
