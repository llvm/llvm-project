# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl weak; weak:' | llvm-mc -filetype=obj -triple=x86_64 - -o %tweak.o
# RUN: echo '.global foo; foo:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: echo '.global bar; bar:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: echo '.global baz; baz:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t3.o
# RUN: rm -f %tweak.a && llvm-ar rc %tweak.a %tweak.o
# RUN: rm -f %t1.a && llvm-ar rc %t1.a %t1.o %t2.o %t3.o

# RUN: ld.lld %t.o %tweak.a %t1.a --print-archive-stats=%t.txt -o /dev/null
# RUN: FileCheck --input-file=%t.txt -DT=%t %s --match-full-lines --strict-whitespace

## Fetches 0 member from %tweak.a and 2 members from %t1.a
#      CHECK:members	fetched	archive
# CHECK-NEXT:1	0	[[T]]weak.a
# CHECK-NEXT:3	2	[[T]]1.a

## - means stdout.
# RUN: ld.lld %t.o %tweak.a %t1.a --print-archive-stats=- -o /dev/null | diff %t.txt -

## The second %t1.a has 0 fetched member.
# RUN: ld.lld %t.o %tweak.a %t1.a %t1.a --print-archive-stats=- -o /dev/null | \
# RUN:   FileCheck --check-prefix=CHECK2 %s
# CHECK2:      members	fetched	archive
# CHECK2-NEXT: 1	0	{{.*}}weak.a
# CHECK2-NEXT: 3	2	{{.*}}1.a
# CHECK2-NEXT: 3	0	{{.*}}1.a

# RUN: not ld.lld -shared %t.o --print-archive-stats=/ -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: --print-archive-stats=: cannot open /: {{.*}}

.globl _start
.weak weak
_start:
  call foo
  call bar
  call weak
