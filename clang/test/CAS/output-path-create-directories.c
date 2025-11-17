// RUN: rm -rf %t %t.cas
// RUN: split-file %s %t

// TODO: Figure out why this fails to create the .i and .dia files.
// XFAIL: *

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -E %t/tu.c -o %t/out/tu.i \
// RUN:   -serialize-diagnostic-file %t/out/tu.dia
// RUN: ls %t/out/B.pcm
// RUN: ls %t/out/B.dia

// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -E %t/tu.c -o %t/out_miss/tu.i \
// RUN:   -serialize-diagnostic-file %t/out_miss/tu.dia \
// RUN:   -fcas-path %t.cas
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache &> %t/B.out.txt
// RUN: cat %t/B.out.txt | FileCheck %s -check-prefix=CACHE-MISS
// RUN: ls %t/out_miss/B.pcm
// RUN: ls %t/out_miss/B.dia

// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -E %t/tu.c -o %t/out_hit/tu.i \
// RUN:   -serialize-diagnostic-file %t/out_hit/B.dia \
// RUN:   -fcas-path %t.cas
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache &> %t/B.out.hit.txt
// RUN: cat %t/B.out.hit.txt | FileCheck %s -check-prefix=CACHE-HIT
// RUN: ls %t/out_hit/B.pcm
// RUN: ls %t/out_hit/B.dia

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- tu.c
