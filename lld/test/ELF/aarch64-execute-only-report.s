// REQUIRES: aarch64

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: llvm-mc --triple=aarch64 --filetype=obj %s -o a.o

// RUN: ld.lld --defsym absolute=0xf0000000 -z execute-only-report=none --fatal-warnings a.o

// RUN: ld.lld --defsym absolute=0xf0000000 -z execute-only-report=warning a.o 2>&1 | \
// RUN:     FileCheck --check-prefix=WARNING %s
// RUN: ld.lld --defsym absolute=0xf0000000 --execute-only -z execute-only-report=warning a.o 2>&1 | \
// RUN:     FileCheck --check-prefix=WARNING %s

// WARNING-NOT: warning: -z execute-only-report: a.o:(.text) does not have SHF_AARCH64_PURECODE flag set
// WARNING-NOT: warning: -z execute-only-report: a.o:(.text.foo) does not have SHF_AARCH64_PURECODE flag set
// WARNING: warning: -z execute-only-report: a.o:(.text.bar) does not have SHF_AARCH64_PURECODE flag set
// WARNING-NOT: warning: -z execute-only-report: <internal>:({{.*}}) does not have SHF_AARCH64_PURECODE flag set

// RUN: not ld.lld --defsym absolute=0xf0000000 -z execute-only-report=error a.o 2>&1 | \
// RUN:     FileCheck --check-prefix=ERROR %s
// RUN: not ld.lld --defsym absolute=0xf0000000 --execute-only -z execute-only-report=error a.o 2>&1 | \
// RUN:     FileCheck --check-prefix=ERROR %s

// ERROR-NOT: error: -z execute-only-report: a.o:(.text) does not have SHF_AARCH64_PURECODE flag set
// ERROR-NOT: error: -z execute-only-report: a.o:(.text.foo) does not have SHF_AARCH64_PURECODE flag set
// ERROR: error: -z execute-only-report: a.o:(.text.bar) does not have SHF_AARCH64_PURECODE flag set
// ERROR-NOT: error: -z execute-only-report: <internal>:({{.*}}) does not have SHF_AARCH64_PURECODE flag set

.section .text,"axy",@progbits,unique,0
.globl _start
_start:
  bl foo
  bl bar
  bl absolute
  ret

.section .text.foo,"axy",@progbits,unique,0
.globl foo
foo:
  ret

.section .text.bar,"ax",@progbits,unique,0
.globl bar
bar:
  ret
