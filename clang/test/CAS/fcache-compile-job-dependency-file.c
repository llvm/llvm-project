// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t
// RUN: llvm-cas --cas %t/cas --ingest --data %t > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o \
// RUN:   -dependency-file %t/deps1.d -MT depends 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
//
// RUN: FileCheck %s --input-file=%t/deps1.d --check-prefix=DEPS
// DEPS: depends:
// DEPS: main.c
// DEPS: my_header.h

// RUN: ls %t/output.o && rm %t/output.o
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %t/main.c -emit-obj -o %t/output.o \
// RUN:   -dependency-file %t/deps2.d -MT depends 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
//
// RUN: ls %t/output.o
// RUN: diff -u %t/deps1.d %t/deps2.d
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit

//--- main.c
#include "my_header.h"

//--- my_header.h
