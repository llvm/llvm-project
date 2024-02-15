# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag1.s -o tag1.o
# RUN: cp tag1.o tag1a.o
# RUN: ld.lld -shared tag1.o tag1a.o -o tagok.so
# RUN: llvm-readelf -n tagok.so | FileCheck --check-prefix OK %s

# OK: AArch64 PAuth ABI tag: platform 0x2a, version 0x1

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag2.s -o tag2.o
# RUN: not ld.lld tag1.o tag1a.o tag2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR1 %s

# ERR1: error: incompatible values of AArch64 PAuth compatibility info found
# ERR1: tag1.o: 0x2A000000000000000{{1|2}}00000000000000
# ERR1: tag2.o: 0x2A000000000000000{{1|2}}00000000000000

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag-errs.s -o errs.o
# RUN: not ld.lld errs.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR2 %s

# ERR2:      error: errs.o:(.note.AARCH64-PAUTH-ABI-tag): invalid type field value 42 (1 expected)
# ERR2-NEXT: error: errs.o:(.note.AARCH64-PAUTH-ABI-tag): invalid name field value XXX (ARM expected)
# ERR2-NEXT: error: errs.o:(.note.AARCH64-PAUTH-ABI-tag): AArch64 PAuth compatibility info is too short (at least 16 bytes expected)

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag-short.s -o short.o
# RUN: not ld.lld short.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR3 %s

# ERR3: error: short.o:(.note.AARCH64-PAUTH-ABI-tag): section is too short

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu no-info.s -o noinfo1.o
# RUN: cp noinfo1.o noinfo2.o
# RUN: not ld.lld -z pauth-report=error tag1.o noinfo1.o noinfo2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR4 %s
# RUN: ld.lld -z pauth-report=warning tag1.o noinfo1.o noinfo2.o -o /dev/null 2>&1 | FileCheck --check-prefix WARN %s
# RUN: ld.lld -z pauth-report=none tag1.o noinfo1.o noinfo2.o --fatal-warnings -o /dev/null

# ERR4:      error: noinfo1.o has no AArch64 PAuth compatibility info while tag1.o has one; either all or no input files must have it
# ERR4-NEXT: error: noinfo2.o has no AArch64 PAuth compatibility info while tag1.o has one; either all or no input files must have it
# WARN:      warning: noinfo1.o has no AArch64 PAuth compatibility info while tag1.o has one; either all or no input files must have it
# WARN-NEXT: warning: noinfo2.o has no AArch64 PAuth compatibility info while tag1.o has one; either all or no input files must have it

#--- abi-tag-short.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 8

#--- abi-tag-errs.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 8
.long 42
.asciz "XXX"

.quad 42

#--- abi-tag1.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 1
.asciz "ARM"

.quad 42         // platform
.quad 1          // version

#--- abi-tag2.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 1
.asciz "ARM"

.quad 42         // platform
.quad 2          // version

#--- no-info.s

.globl _start;   // define _start to avoid missing entry warning and use --fatal-warnings to assert no diagnostic
.weak _start;    // allow multiple definitions of _start for simplicity
_start:

.section ".test", "a"
