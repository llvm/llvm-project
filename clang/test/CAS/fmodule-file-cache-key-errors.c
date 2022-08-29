// Checks error conditions related to -fmodule-file-cache-key, e.g. missing
// cache entry, invalid id, etc.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: llvm-cas --cas %t/cas --ingest --data %t > %t/casid

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file-cache-key=INVALID \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/invalid.txt
// RUN: cat %t/invalid.txt | FileCheck %s -check-prefix=INVALID

// INVALID: error: option '-fmodule-file-cache-key' should be of the form <path>=<key>

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file-cache-key=PATH=KEY \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/bad_key.txt
// RUN: cat %t/bad_key.txt | FileCheck %s -check-prefix=BAD_KEY

// BAD_KEY: error: CAS cannot load module with key 'KEY' from -fmodule-file-cache-key: invalid cas-id 'KEY'

// RUN: echo -n '-fmodule-file-cache-key=PATH=' > %t/not_in_cache.rsp
// RUN: cat %t/casid >> %t/not_in_cache.rsp

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/not_in_cache.rsp \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/not_in_cache.txt
// RUN: cat %t/not_in_cache.txt | FileCheck %s -check-prefix=NOT_IN_CACHE

// NOT_IN_CACHE: error: CAS cannot load module with key '{{.*}}' from -fmodule-file-cache-key: no such entry in action cache

//--- module.modulemap
module A { header "A.h" }

//--- A.h
void A(void);

//--- tu.c
#include "A.h"
void tu(void) {
  A();
}
