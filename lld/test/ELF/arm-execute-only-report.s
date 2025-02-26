// REQUIRES: arm

// RUN: llvm-mc --triple=armv7 --filetype=obj -o %t.o %s
// RUN: ld.lld --defsym absolute=0xf0000000 -z execute-only-report=none --fatal-warnings %t.o -o /dev/null
// RUN: ld.lld --defsym absolute=0xf0000000 -z execute-only-report=warning %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck --check-prefix=WARNING %s
// RUN: not ld.lld --defsym absolute=0xf0000000 -z execute-only-report=error %t.o -o /dev/null 2>&1 \
// RUN:     | FileCheck --check-prefix=ERROR %s

// WARNING-NOT: warning: -z execute-only-report: {{.*}}.o:(.text) does not have SHF_ARM_PURECODE flag set
// WARNING-NOT: warning: -z execute-only-report: {{.*}}.o:(.text.foo) does not have SHF_ARM_PURECODE flag set
// WARNING: warning: -z execute-only-report: {{.*}}.o:(.text.bar) does not have SHF_ARM_PURECODE flag set
// WARNING-NOT: warning: -z execute-only-report: <internal>:({{.*}}) does not have SHF_ARM_PURECODE flag set

// ERROR-NOT: error: -z execute-only-report: {{.*}}.o:(.text) does not have SHF_ARM_PURECODE flag set
// ERROR-NOT: error: -z execute-only-report: {{.*}}.o:(.text.foo) does not have SHF_ARM_PURECODE flag set
// ERROR: error: -z execute-only-report: {{.*}}.o:(.text.bar) does not have SHF_ARM_PURECODE flag set
// ERROR-NOT: error: -z execute-only-report: <internal>:({{.*}}) does not have SHF_ARM_PURECODE flag set

.section .text,"axy",%progbits,unique,0
.globl _start
_start:
  bl foo
  bl bar
  bl absolute
  bx lr

.section .text.foo,"axy",%progbits,unique,0
.globl foo
foo:
  bx lr

.section .text.bar,"ax",%progbits,unique,0
.globl bar
bar:
  bx lr
