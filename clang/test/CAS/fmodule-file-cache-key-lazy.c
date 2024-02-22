// Tests for combining -fmodule-file-cache-key with lazy-loading modules via
// -fmodule-file=<NAME>=<PATH>.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t %t.cas
// RUN: split-file %s %t

// RUN: llvm-cas --cas %t.cas --ingest %t > %t/casid

// == Build B

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=B -fno-implicit-modules \
// RUN:   -emit-module %t/module.modulemap -o %t/B.pcm \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/B.out.txt
// RUN: cat %t/B.out.txt | FileCheck %s --check-prefix=CACHE-MISS
// RUN: cat %t/B.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/B.key

// == Build A, importing B

// RUN: echo -n '-fmodule-file-cache-key %t/B.pcm ' > %t/B.import.rsp
// RUN: cat %t/B.key >> %t/B.import.rsp

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=A -fno-implicit-modules \
// RUN:   @%t/B.import.rsp -fmodule-file=B=%t/B.pcm \
// RUN:   -emit-module %t/module.modulemap -o %t/A.pcm \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/A.out.txt
// RUN: cat %t/A.out.txt | FileCheck %s --check-prefix=CACHE-MISS
// RUN: cat %t/A.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/A.key

// == Build tu, importing A (implicitly importing B)

// RUN: echo -n '-fmodule-file-cache-key %t/A.pcm ' > %t/A.import.rsp
// RUN: cat %t/A.key >> %t/A.import.rsp

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/A.import.rsp -fmodule-file=A=%t/A.pcm \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/tu.out.txt
// RUN: cat %t/tu.out.txt | FileCheck %s --check-prefix=CACHE-MISS

// == Ensure we're reading pcm from cache

// RUN: rm %t/*.pcm

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/A.import.rsp -fmodule-file=A=%t/A.pcm \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/tu.out.2.txt
// RUN: cat %t/tu.out.2.txt | FileCheck %s --check-prefix=CACHE-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- module.modulemap
module A { header "A.h" export * }
module B { header "B.h" }

//--- A.h
#include "B.h"

//--- B.h
void B(void);

//--- tu.c
#include "A.h"
void tu(void) {
  B();
}
