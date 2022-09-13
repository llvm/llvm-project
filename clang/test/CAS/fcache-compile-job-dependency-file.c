// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t
// RUN: llvm-cas --cas %t/cas --ingest --data %t > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:   -dependency-file %t/deps1.d -MT depends 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
//
// RUN: FileCheck %s --input-file=%t/deps1.d --check-prefix=DEPS
// DEPS: depends:
// DEPS: main.c
// DEPS: my_header.h
// DEPS-NOT: sys.h

// RUN: ls %t/output.o && rm %t/output.o
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:   -dependency-file %t/deps2.d -MT depends 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
//
// RUN: ls %t/output.o
// RUN: diff -u %t/deps1.d %t/deps2.d
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit

// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:   -dependency-file %t/deps3.d -MT other1 -MT other2 -MP -fdepfile-entry=extra-depfile.json 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT

// RUN: FileCheck %s --input-file=%t/deps3.d --check-prefix=DEPS_OTHER
// DEPS_OTHER: other1 other2:
// DEPS_OTHER: extra-depfile.json
// DEPS_OTHER: main.c
// DEPS_OTHER: my_header.h
// DEPS_OTHER-NOT: sys.h
// DEPS_OTHER: my_header.h:

// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:   -sys-header-deps -dependency-file %t/deps4.d -MT depends 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-MISS

// Note: currently options that affect the list of deps (like sys-header-deps)
// are part of the cache key, to avoid saving unnecessary paths.

// RUN: FileCheck %s --input-file=%t/deps4.d --check-prefix=DEPS_SYS
// DEPS_SYS: depends:
// DEPS_SYS: main.c
// DEPS_SYS: my_header.h
// DEPS_SYS: sys.h

// Using another cache path to avoid reusing artifacts.

// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache2 -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:   -dependency-file %t/deps-depfile1.d -MT deps -fdepfile-entry=extra-depfile.json -fdepfile-entry=%t/main.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-MISS

// RUN: FileCheck %s --input-file=%t/deps-depfile1.d --check-prefix=DEPS_DEPFILE1
// DEPS_DEPFILE1: deps:
// DEPS_DEPFILE1: extra-depfile.json
// DEPS_DEPFILE1: main.c

// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache2 -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o -isystem %t/sys \
// RUN:   -dependency-file %t/deps-depfile2.d -MT deps 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT

// RUN: FileCheck %s --input-file=%t/deps-depfile2.d --check-prefix=DEPS_DEPFILE2
// DEPS_DEPFILE2-NOT: extra-depfile.json
// DEPS_DEPFILE2: main.c

//--- main.c
#include "my_header.h"
#include <sys.h>

//--- my_header.h

//--- sys/sys.h
