# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null 2>&1 --warn-32-bit-reloc-to-large-section | FileCheck %s
# RUN: ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO --allow-empty

# CHECK: warning: {{.*}}warn-large.s{{.*}}:(.text+{{.*}}): Large section should not be addressed with PC32 relocation; references 'hello'
# CHECK-NEXT: >>> referenced by foo.c
# CHECK-NEXT: >>> defined in {{.*}}warn-large.s{{.*}}

# CHECK: warning: {{.*}}warn-large.s{{.*}}:(.text+{{.*}}): Large section should not be addressed with PC32 relocation; references section 'ldata'
# CHECK-NEXT: >>> referenced by foo.c

# NO-NOT: warning

.text
.file "foo.c"
.globl _start
.type _start, @function
_start:
  movq hello(%rip), %rax
  movq ok(%rip), %rax

.section ldata,"awl",@progbits

.type   hello, @object
.globl  hello
.p2align        2, 0x0
hello:
.long   1

ok:
.long   1
