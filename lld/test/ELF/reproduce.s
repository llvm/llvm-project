# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build1/foo.o
# RUN: cd %t.dir
# RUN: ld.lld --hash-style=gnu build1/foo.o -o bar -shared --as-needed --reproduce repro
# RUN: diff build1/foo.o repro/%:t.dir/build1/foo.o

# RUN: FileCheck %s --check-prefix=RSP < repro/response.txt
# RSP: {{^}}--hash-style gnu{{$}}
# RSP-NOT: repro/
# RSP-NEXT: /foo.o
# RSP-NEXT: -o bar
# RSP-NEXT: -shared
# RSP-NEXT: --as-needed

# RUN: mkdir -p %t.dir/build2/a/b/c
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build2/foo.o
# RUN: cd %t.dir/build2/a/b/c
# RUN: ld.lld ./../../../foo.o -o bar -shared --as-needed --reproduce repro
# RUN: diff %t.dir/build2/foo.o repro/%:t.dir/build2/foo.o

# RUN: touch file
# RUN: not ld.lld --reproduce repro2 'foo bar' -L"foo bar" -Lfile -version-script file
# RUN: FileCheck %s --check-prefix=RSP2 < repro2/response.txt
# RSP2:      "foo bar"
# RSP2-NEXT: -L "foo bar"
# RSP2-NEXT: -L {{.+}}file
# RSP2-NEXT: -version-script {{.+}}file

# RUN: not ld.lld build1/foo.o -o bar -shared --as-needed --reproduce . 2>&1 \
# RUN:   | FileCheck --check-prefix=ERROR %s
# ERROR: can't create directory

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
