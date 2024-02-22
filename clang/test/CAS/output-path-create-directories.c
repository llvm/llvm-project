// RUN: rm -rf %t %t.cas
// RUN: split-file %s %t
// RUN: llvm-cas --cas %t.cas --ingest %t > %t/casid

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=Mod -fno-implicit-modules \
// RUN:   -emit-module %t/module.modulemap -o %t/out/B.pcm \
// RUN:   -serialize-diagnostic-file %t/out/B.dia
// RUN: ls %t/out/B.pcm
// RUN: ls %t/out/B.dia

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=Mod -fno-implicit-modules \
// RUN:   -emit-module %t/module.modulemap -o %t/out_miss/B.pcm \
// RUN:   -serialize-diagnostic-file %t/out_miss/B.dia \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/B.out.txt
// RUN: cat %t/B.out.txt | FileCheck %s -check-prefix=CACHE-MISS
// RUN: ls %t/out_miss/B.pcm
// RUN: ls %t/out_miss/B.dia

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=Mod -fno-implicit-modules \
// RUN:   -emit-module %t/module.modulemap -o %t/out_hit/B.pcm \
// RUN:   -serialize-diagnostic-file %t/out_hit/B.dia \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/B.out.hit.txt
// RUN: cat %t/B.out.hit.txt | FileCheck %s -check-prefix=CACHE-HIT
// RUN: ls %t/out_hit/B.pcm
// RUN: ls %t/out_hit/B.dia

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- module.modulemap
module Mod { header "Header.h" }

//--- Header.h
