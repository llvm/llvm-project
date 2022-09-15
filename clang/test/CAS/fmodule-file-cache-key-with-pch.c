// Check that -fmodule-file-cache-key works with mixed PCH+modules builds.
// This test mimics the way the dep scanner handles PCH (ie. treat it as a file
// input, ingested into the cas fs).

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: llvm-cas --cas %t/cas --ingest --data %t > %t/casid

// == Build B

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=B -fno-implicit-modules \
// RUN:   -fmodule-related-to-pch \
// RUN:   -emit-module %t/module.modulemap -o %t/B.pcm \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/B.out.txt
// RUN: cat %t/B.out.txt | FileCheck %s --check-prefix=CACHE-MISS
// RUN: cat %t/B.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/B.key

// == Build A, importing B

// RUN: echo -n '-fmodule-file-cache-key=%t/B.pcm=' > %t/B.import.rsp
// RUN: cat %t/B.key >> %t/B.import.rsp

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=A -fno-implicit-modules \
// RUN:   -fmodule-related-to-pch \
// RUN:   @%t/B.import.rsp -fmodule-file=%t/B.pcm \
// RUN:   -emit-module %t/module.modulemap -o %t/A.pcm \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/A.out.txt
// RUN: cat %t/A.out.txt | FileCheck %s --check-prefix=CACHE-MISS
// RUN: cat %t/A.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/A.key

// == Build C, importing B

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=C -fno-implicit-modules \
// RUN:   -fmodule-related-to-pch \
// RUN:   @%t/B.import.rsp -fmodule-file=%t/B.pcm \
// RUN:   -emit-module %t/module.modulemap -o %t/C.pcm \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/C.out.txt
// RUN: cat %t/C.out.txt | FileCheck %s --check-prefix=CACHE-MISS
// RUN: cat %t/C.out.txt | sed -E "s:^.*cache [a-z]+ for '([^']+)'.*$:\1:" > %t/C.key

// == Build PCH, importing A (implicitly importing B)

// RUN: echo -n '-fmodule-file-cache-key=%t/A.pcm=' > %t/A.import.rsp
// RUN: cat %t/A.key >> %t/A.import.rsp

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/A.import.rsp -fmodule-file=%t/A.pcm \
// RUN:   -emit-pch -x c-header %t/prefix.h -o %t/prefix.pch \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/prefix.out.txt
// RUN: cat %t/prefix.out.txt | FileCheck %s --check-prefix=CACHE-MISS

// == Clear pcms to ensure they load from cache, and re-ingest with pch

// RUN: rm %t/*.pcm
// RUN: llvm-cas --cas %t/cas --ingest --data %t > %t/casid
// RUN: rm %t/*.pch

// == Build tu

// RUN: echo -n '-fmodule-file-cache-key=%t/C.pcm=' > %t/C.import.rsp
// RUN: cat %t/C.key >> %t/C.import.rsp

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/C.import.rsp -fmodule-file=%t/C.pcm -include-pch %t/prefix.pch \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/tu.out.txt
// RUN: cat %t/tu.out.txt | FileCheck %s --check-prefix=CACHE-MISS

// == Ensure we're reading pcm from cache

// RUN: %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fno-implicit-modules \
// RUN:   @%t/C.import.rsp -fmodule-file=%t/C.pcm -include-pch %t/prefix.pch \
// RUN:   -fsyntax-only %t/tu.c \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/tu.out.2.txt
// RUN: cat %t/tu.out.2.txt | FileCheck %s --check-prefix=CACHE-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- module.modulemap
module A { header "A.h" export * }
module B { header "B.h" }
module C { header "C.h" export * }

//--- A.h
#include "B.h"

//--- B.h
void B(void);

//--- C.h
#include "B.h"
void B(void);

//--- prefix.h
#include "A.h"

//--- tu.c
#include "C.h"
void tu(void) {
  B();
}
