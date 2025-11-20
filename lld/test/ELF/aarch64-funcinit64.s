# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s -r %t | FileCheck %s
# RUN: ld.lld %t.o -o %t -pie
# RUN: llvm-readelf -s -r %t | FileCheck %s
# RUN: not ld.lld %t.o -o %t -shared 2>&1 | FileCheck --check-prefix=ERR %s

.data
# CHECK: R_AARCH64_IRELATIVE [[FOO:[0-9a-f]*]]
# ERR: relocation R_AARCH64_FUNCINIT64 cannot be used against preemptible symbol 'foo'
.8byte foo@FUNCINIT

.text
# CHECK: {{0*}}[[FOO]] {{.*}} foo
.globl foo
foo:
ret
