# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: echo '.globl weak; weak:' | llvm-mc -filetype=obj -triple=x86_64 - -o weak.o
# RUN: echo '.global foo; foo:' | llvm-mc -filetype=obj -triple=x86_64 - -o 1.o
# RUN: echo '.global bar; bar:' | llvm-mc -filetype=obj -triple=x86_64 - -o 2.o
# RUN: echo '.global baz; baz:' | llvm-mc -filetype=obj -triple=x86_64 - -o 3.o
# RUN: rm -f weak.a && llvm-ar rc weak.a weak.o
# RUN: rm -f 1.a && llvm-ar rc 1.a 1.o 2.o 3.o

# RUN: ld.lld a.o %t/weak.a 1.a --print-archive-stats=a.txt -o /dev/null
# RUN: FileCheck --input-file=a.txt -DT=%t %s --match-full-lines --strict-whitespace

## Fetches 0 member from %t/weak.a and 2 members from %t1.a
#      CHECK:members	extracted	archive
# CHECK-NEXT:1	0	[[T]]/weak.a
# CHECK-NEXT:3	2	1.a

## - means stdout.
# RUN: ld.lld a.o %t/weak.a 1.a --print-archive-stats=- -o /dev/null | diff a.txt -

## The second 1.a has 0 fetched member.
# RUN: ld.lld a.o %t/weak.a -L. -l:1.a -l:1.a --print-archive-stats=- -o /dev/null | \
# RUN:   FileCheck --check-prefix=CHECK2 %s
# CHECK2:      members	extracted	archive
# CHECK2-NEXT: 1	0	{{.*}}weak.a
# CHECK2-NEXT: 3	2	{{.*}}1.a
# CHECK2-NEXT: 3	0	{{.*}}1.a

# RUN: not ld.lld -shared a.o --print-archive-stats=/ -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: --print-archive-stats=: cannot open /: {{.*}}

#--- a.s
.globl _start
.weak weak
_start:
  call foo
  call bar
  call weak
