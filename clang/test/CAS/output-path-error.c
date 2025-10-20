// Check that fatal errors from cache-related output paths show up.

// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest %s > %t/casid

// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS

// Remove only the CAS, but leave the ActionCache.
// RUN: rm -rf %t/cas

// RUN: not %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o &> %t/output.txt
// RUN: cat %t/output.txt | FileCheck %s --check-prefix=ERROR

// CACHE-MISS: remark: compile job cache miss
// ERROR: fatal error: CAS filesystem cannot be initialized from root-id 'llvmcas://{{.*}}': cannot get reference to root FS
