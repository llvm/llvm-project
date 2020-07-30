# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK: 0 NOTYPE GLOBAL PROTECTED 2 __start_aaa
# CHECK: 0 NOTYPE GLOBAL PROTECTED 2 __stop_aaa

# RUN: ld.lld -z start-stop-visibility=default %t.o -o %t1
# RUN: llvm-readelf -s %t1 | FileCheck --check-prefix=CHECK-DEFAULT %s

# CHECK-DEFAULT: 0 NOTYPE GLOBAL DEFAULT 2 __start_aaa
# CHECK-DEFAULT: 0 NOTYPE GLOBAL DEFAULT 2 __stop_aaa

# RUN: ld.lld -z start-stop-visibility=internal %t.o -o %t2
# RUN: llvm-readelf -s %t2 | FileCheck --check-prefix=CHECK-INTERNAL %s

# CHECK-INTERNAL: 0 NOTYPE LOCAL INTERNAL 2 __start_aaa
# CHECK-INTERNAL: 0 NOTYPE LOCAL INTERNAL 2 __stop_aaa

# RUN: ld.lld -z start-stop-visibility=hidden %t.o -o %t3
# RUN: llvm-readelf -s %t3 | FileCheck --check-prefix=CHECK-HIDDEN %s

# CHECK-HIDDEN: 0 NOTYPE LOCAL HIDDEN 2 __start_aaa
# CHECK-HIDDEN: 0 NOTYPE LOCAL HIDDEN 2 __stop_aaa

# RUN: ld.lld -z start-stop-visibility=protected %t.o -o %t4
# RUN: llvm-readelf -s %t4 | FileCheck --check-prefix=CHECK-PROTECTED %s

# CHECK-PROTECTED: 0 NOTYPE GLOBAL PROTECTED 2 __start_aaa
# CHECK-PROTECTED: 0 NOTYPE GLOBAL PROTECTED 2 __stop_aaa

# RUN: not ld.lld -z start-stop-visibility=aaa %t.o -o /dev/null
# CHECK-ERROR: error: unknown -z start-stop-visibility= value: aaa

.quad __start_aaa
.quad __stop_aaa
.section aaa,"ax"

.global _start
.text
_start:
  nop
