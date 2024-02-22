// Checks error conditions related to -fmodule-file-cache-key, e.g. missing
// cache entry, invalid id, etc.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t %t.cas %t.cas_2
// RUN: split-file %s %t

// RUN: llvm-cas --cas %t.cas --ingest %t > %t/casid

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file-cache-key=INVALID \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/invalid.txt
// RUN: cat %t/invalid.txt | FileCheck %s -check-prefix=INVALID

// INVALID: error: unknown argument: '-fmodule-file-cache-key=INVALID'

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file-cache-key INVALID \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/invalid2.txt
// RUN: FileCheck %s -check-prefix=INVALID2 -input-file=%t/invalid2.txt

// INVALID2: error: CAS cannot load module with key '-fsyntax-only' from -fmodule-file-cache-key

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file-cache-key INVALID ALSO_INVALID MORE_INVALID \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/invalid3.txt
// RUN: FileCheck %s -check-prefix=INVALID3 -input-file=%t/invalid3.txt

// INVALID3: error: error reading 'MORE_INVALID'

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file-cache-key PATH KEY \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/bad_key.txt
// RUN: cat %t/bad_key.txt | FileCheck %s -check-prefix=BAD_KEY

// BAD_KEY: error: CAS cannot load module with key 'KEY' from -fmodule-file-cache-key: invalid cas-id 'KEY'

// RUN: echo -n '-fmodule-file-cache-key PATH ' > %t/bad_key2.rsp
// RUN: cat %t/casid >> %t/bad_key2.rsp

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/bad_key2.rsp \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/bad_key2.txt
// RUN: cat %t/bad_key2.txt | FileCheck %s -check-prefix=BAD_KEY2

// BAD_KEY2: error: CAS cannot load module with key '{{.*}}' from -fmodule-file-cache-key: cas object is not a valid cache key

// == Build A

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=A -fno-implicit-modules \
// RUN:   -emit-module %t/module.modulemap -o %t/A.pcm \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/A.out.txt
// RUN: cat %t/A.out.txt | FileCheck %s --check-prefix=CACHE-MISS
// CACHE-MISS: remark: compile job cache miss
// RUN: cat %t/A.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/A.key

// == Try to import A with an empty action cache, simulating a missing module

// RUN: llvm-cas --cas %t.cas_2 --import --upstream-cas %t.cas @%t/A.key

// RUN: echo -n '-fmodule-file-cache-key PATH ' > %t/not_in_cache.rsp
// RUN: cat %t/A.key >> %t/not_in_cache.rsp

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/not_in_cache.rsp \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas_2 -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/not_in_cache.txt
// RUN: cat %t/not_in_cache.txt | FileCheck %s -check-prefix=NOT_IN_CACHE -DPREFIX=%/t

// NOT_IN_CACHE: error: CAS cannot load module with key '{{.*}}' from -fmodule-file-cache-key: no such entry in action cache; expected compile:
// NOT_IN_CACHE: command-line:
// NOT_IN_CACHE:   -cc1
// NOT_IN_CACHE: filesystem:
// NOT_IN_CACHE:   file llvmcas://{{.*}} [[PREFIX]]/A.h

//--- module.modulemap
module A { header "A.h" }

//--- A.h
void A(void);

//--- tu.c
#include "A.h"
void tu(void) {
  A();
}
