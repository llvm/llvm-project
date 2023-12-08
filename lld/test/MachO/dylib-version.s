# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: not %lld -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 10 -current_version 11 2>&1 | \
# RUN:     FileCheck --check-prefix=NEEDDYLIB %s
# RUN: not %lld -execute -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 10 -current_version 11 2>&1 | \
# RUN:     FileCheck --check-prefix=NEEDDYLIB %s
# RUN: not %lld -bundle -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 10 -current_version 11 2>&1 | \
# RUN:     FileCheck --check-prefix=NEEDDYLIB %s

# NEEDDYLIB: error: -compatibility_version 10: only valid with -dylib
# NEEDDYLIB: error: -current_version 11: only valid with -dylib

# RUN: not %lld -dylib -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 1.2.3.4 -current_version 1.2.3.4.5 2>&1 | \
# RUN:     FileCheck --check-prefix=MALFORMED %s

# MALFORMED: error: -compatibility_version 1.2.3.4: malformed version
# MALFORMED: error: -current_version 1.2.3.4.5: malformed version

# RUN: not %lld -dylib -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 80000.1 -current_version 1.2.3 2>&1 | \
# RUN:     FileCheck --check-prefix=BADMAJOR %s

# BADMAJOR: error: -compatibility_version 80000.1: malformed version

# RUN: not %lld -dylib -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 8.300 -current_version 1.2.3 2>&1 | \
# RUN:     FileCheck --check-prefix=BADMINOR %s

# BADMINOR: error: -compatibility_version 8.300: malformed version

# RUN: not %lld -dylib -o %t/executable %t.o -o /dev/null \
# RUN:     -compatibility_version 8.8.300 -current_version 1.2.3 2>&1 | \
# RUN:     FileCheck --check-prefix=BADSUBMINOR %s

# BADSUBMINOR: error: -compatibility_version 8.8.300: malformed version

# RUN: %lld -dylib -o %t/executable %t.o -o %t.dylib \
# RUN:     -compatibility_version 1.2.3 -current_version 2.5.6
# RUN: llvm-objdump --macho --all-headers %t.dylib | FileCheck %s

# CHECK:      cmd LC_ID_DYLIB
# CHECK-NEXT: cmdsize
# CHECK-NEXT: name
# CHECK-NEXT: time stamp
# CHECK-NEXT: current version 2.5.6
# CHECK-NEXT: compatibility version 1.2.3

.text
.global _foo
_foo:
  mov $0, %rax
  ret
