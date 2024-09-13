// Tests for reusing a PCM output from CAS builds by a non-cas build.
// This is to simulate a configuration by debugger without CAS support.

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
// RUN: cat %t/B.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/B.key

// == Build A, importing B

// RUN: echo -n '-fmodule-file-cache-key %t/B.pcm ' > %t/B.import.rsp
// RUN: cat %t/B.key >> %t/B.import.rsp

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=A -fno-implicit-modules \
// RUN:   @%t/B.import.rsp -fmodule-file=%t/B.pcm \
// RUN:   -emit-module %t/module.modulemap -o %t/A.pcm \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/A.out.txt
// RUN: cat %t/A.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/A.key

// == Build tu, importing A and B, without a CAS, this should fail.

// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file=%t/A.pcm\
// RUN:   -fmodule-file=%t/B.pcm\
// RUN:   -fsyntax-only %t/tu.c

// == Using option to ignore CAS info inside module

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   -fmodule-file=%t/A.pcm\
// RUN:   -fmodule-file=%t/B.pcm\
// RUN:   -fsyntax-only %t/tu.c -fmodule-load-ignore-cas

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
