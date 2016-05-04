# REQUIRES: x86

# Extracting the cpio archive can get over the path limit on windows.
# REQUIRES: shell

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build1/foo.o
# RUN: cd %t.dir
# RUN: ld.lld --hash-style=gnu build1/foo.o -o bar -shared --as-needed --reproduce repro
# RUN: cpio -id < repro.cpio
# RUN: diff build1/foo.o repro/%:t.dir/build1/foo.o

# RUN: FileCheck %s --check-prefix=RSP < repro/response.txt
# RSP: {{^}}--hash-style gnu{{$}}
# RSP-NOT: repro{{[/\\]}}
# RSP-NEXT: {{[/\\]}}foo.o
# RSP-NEXT: -o bar
# RSP-NEXT: -shared
# RSP-NEXT: --as-needed

# RUN: mkdir -p %t.dir/build2/a/b/c
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build2/foo.o
# RUN: cd %t.dir/build2/a/b/c
# RUN: ld.lld ./../../../foo.o -o bar -shared --as-needed --reproduce repro
# RUN: cpio -id < repro.cpio
# RUN: diff %t.dir/build2/foo.o repro/%:t.dir/build2/foo.o

# RUN: echo "{ local: *; };" >  ver
# RUN: echo > dyn
# RUN: echo > file
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o 'foo bar'
# RUN: ld.lld --reproduce repro2 'foo bar' -L"foo bar" -Lfile \
# RUN:   --dynamic-list dyn -rpath file --script file --version-script ver \
# RUN:   --dynamic-linker "some unusual/path"
# RUN: cpio -id < repro2.cpio
# RUN: FileCheck %s --check-prefix=RSP2 < repro2/response.txt
# RSP2:      "{{.*}}foo bar"
# RSP2-NEXT: -L "{{.*}}foo bar"
# RSP2-NEXT: -L {{.+}}file
# RSP2-NEXT: --dynamic-list {{.+}}dyn
# RSP2-NEXT: -rpath {{.+}}file
# RSP2-NEXT: --script {{.+}}file
# RSP2-NEXT: --version-script {{.+}}ver
# RSP2-NEXT: --dynamic-linker "some unusual/path"

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
