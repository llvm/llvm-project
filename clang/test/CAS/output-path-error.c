// Check that fatal errors from cache-related output paths show up.

// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o
// RUN: %clang @%t.rsp 2>&1 | FileCheck %s --allow-empty --check-prefix=CACHE-MISS

// Remove only the CAS, but leave the ActionCache.
// RUN: rm -rf %t/cas

// RUN: not %clang @%t.rsp 2>&1 | FileCheck %s --allow-empty --check-prefix=ERROR

// CACHE-MISS: remark: compile job cache miss
// ERROR: fatal error: CAS filesystem cannot be initialized from root-id 'llvmcas://{{.*}}': include-tree CASID does not exist
