# REQUIRES: x86
## Test symbol resolution related to .symver produced symbols in object files.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/ref.s -o %t/ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/ref1.s -o %t/ref1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/ref1p.s -o %t/ref1p.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/ref1w.s -o %t/ref1w.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def1.s -o %t/def1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def1w.s -o %t/def1w.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/hid1.s -o %t/hid1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/hid1w.s -o %t/hid1w.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def2.s -o %t/def2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/wrap.s -o %t/wrap.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/wrap1.s -o %t/wrap1.o
# RUN: ld.lld -shared --soname=def1.so --version-script=%t/ver %t/def1.o -o %t/def1.so
# RUN: ld.lld -shared --soname=hid1.so --version-script=%t/ver %t/hid1.o -o %t/hid1.so

## Report a duplicate definition error for foo@v1 and foo@@v1.
# RUN: not ld.lld -shared --version-script=%t/ver %t/def1.o %t/hid1.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DUP
# RUN: ld.lld -shared --version-script=%t/ver %t/def1w.o %t/hid1.o -o /dev/null
# RUN: ld.lld -shared --version-script=%t/ver %t/def1.o %t/hid1w.o -o /dev/null
# RUN: ld.lld -shared --version-script=%t/ver %t/def1w.o %t/hid1w.o -o /dev/null

# DUP:      error: duplicate symbol: foo@@v1
# DUP-NEXT: >>> defined at {{.*}}/def1{{w?}}.o:(.text+0x0)
# DUP-NEXT: >>> defined at {{.*}}/hid1.o:(.text+0x0)

## Protected undefined foo@v1 makes the output symbol protected.
# RUN: ld.lld -shared --version-script=%t/ver %t/ref1p.o %t/def1.o -o %t.protected
# RUN: llvm-readelf --dyn-syms %t.protected | FileCheck %s --check-prefix=PROTECTED

# PROTECTED:  NOTYPE GLOBAL PROTECTED [[#]] foo@@v1

## foo@@v1 resolves both undefined foo and foo@v1. There is one single definition.
## Note: set soname so that the name string referenced by .gnu.version_d is fixed.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver %t/ref.o %t/ref1.o %t/def1.o -o %t1
# RUN: llvm-readelf -r --dyn-syms %t1 | FileCheck %s

# CHECK:       Relocation section '.rela.plt' at offset {{.*}} contains 1 entries:
# CHECK-NEXT:  {{.*}} Type               {{.*}}
# CHECK-NEXT:  {{.*}} R_X86_64_JUMP_SLOT {{.*}} foo@@v1 + 0

# CHECK:       1: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@@v1
# CHECK-EMPTY:

## foo@@v2 does not resolve undefined foo@v1.
# RUN: not ld.lld -shared --soname=t --version-script=%t/ver %t/ref.o %t/ref1.o %t/def2.o \
# RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEF

# UNDEF: error: undefined symbol: foo@v1

## An undefined weak unversioned symbol is not errored. However, an undefined
## weak versioned symbol should still be errored because we cannot construct
## a Verneed entry (Verneed::vn_file is unavailable).
# RUN: not ld.lld -shared --version-script=%t/ver %t/ref1w.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=UNDEF

## foo@@v2 resolves undefined foo while foo@v1 resolves undefined foo@v1.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver %t/ref.o %t/ref1.o %t/hid1.o %t/def2.o -o %t3
# RUN: llvm-readelf -r --dyn-syms %t3 | FileCheck %s --check-prefix=CHECK3
# RUN: llvm-objdump -d --no-show-raw-insn %t3 | FileCheck %s --check-prefix=DIS3

# CHECK3:       00000000000034a8 {{.*}} R_X86_64_JUMP_SLOT {{.*}} foo@@v2 + 0
# CHECK3-NEXT:  00000000000034b0 {{.*}} R_X86_64_JUMP_SLOT {{.*}} foo@v1 + 0

# CHECK3:       1: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@@v2
# CHECK3-NEXT:  2: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@v1
# CHECK3-NEXT:  3: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo_v1
# CHECK3-EMPTY:

# DIS3-LABEL: <.text>:
# DIS3-NEXT:    callq 0x1380 <foo@plt>
# DIS3-COUNT-3: int3
# DIS3-NEXT:    callq 0x1390 <foo@plt>
# DIS3-LABEL: <foo@plt>:
# DIS3-NEXT:    jmpq *{{.*}}(%rip) # 0x34a8
# DIS3-LABEL: <foo@plt>:
# DIS3-NEXT:    jmpq *{{.*}}(%rip) # 0x34b0

## Then, test the interaction with versioned definitions in shared objects.

## TODO Both foo and foo@v1 are undefined. Ideally we should not create two .dynsym entries.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver %t/ref.o %t/ref1.o %t/def1.so -o %t4
# RUN: llvm-readelf --dyn-syms %t4 | FileCheck %s --check-prefix=CHECK4

# CHECK4:       1: {{.*}} NOTYPE GLOBAL DEFAULT UND   foo@v1
# CHECK4-NEXT:  2: {{.*}} NOTYPE GLOBAL DEFAULT UND   foo@v1
# CHECK4-EMPTY:

## hid1.so resolves undefined foo@v1. foo is undefined.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver %t/ref.o %t/ref1.o %t/hid1.so -o %t5
# RUN: llvm-readelf --dyn-syms %t5 | FileCheck %s --check-prefix=CHECK5

# CHECK5:       1: {{.*}} NOTYPE GLOBAL DEFAULT UND   foo{{$}}
# CHECK5-NEXT:  2: {{.*}} NOTYPE GLOBAL DEFAULT UND   foo@v1
# CHECK5-EMPTY:

## Test the interaction with --wrap.

## The reference from ref.o is redirected. The reference from ref1.o is not.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver --wrap=foo %t/ref.o %t/ref1.o %t/def1.o %t/wrap.o -o %t.w1
# RUN: llvm-readobj -r %t.w1 | FileCheck %s --check-prefix=W1REL
# RUN: llvm-objdump -d --no-show-raw-insn %t.w1 | FileCheck %s --check-prefix=W1DIS

# W1REL:      .rela.plt {
# W1REL-NEXT:   R_X86_64_JUMP_SLOT __wrap_foo 0x0
# W1REL-NEXT:   R_X86_64_JUMP_SLOT foo@@v1 0x0
# W1REL-NEXT: }

# W1DIS-LABEL: <.text>:
# W1DIS-NEXT:    callq {{.*}} <__wrap_foo@plt>
# W1DIS-COUNT-3: int3
# W1DIS-NEXT:    callq {{.*}} <foo@plt>

## The reference from ref.o is redirected. The reference from ref1.o is not.
## Note: this case demonstrates the typical behavior wrapping a glibc libc.so definition.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver --wrap=foo %t/ref.o %t/ref1.o %t/def1.so %t/wrap.o -o %t.w2
# RUN: llvm-readobj -r %t.w2 | FileCheck %s --check-prefix=W2REL
# RUN: llvm-objdump -d --no-show-raw-insn %t.w2 | FileCheck %s --check-prefix=W2DIS

# W2REL:      .rela.plt {
# W2REL-NEXT:   R_X86_64_JUMP_SLOT __wrap_foo 0x0
# W2REL-NEXT:   R_X86_64_JUMP_SLOT foo@v1 0x0
# W2REL-NEXT: }

# W2DIS-LABEL: <.text>:
# W2DIS-NEXT:    callq {{.*}} <__wrap_foo@plt>
# W2DIS-COUNT-3: int3
# W2DIS-NEXT:    callq {{.*}} <foo@plt>

## Test --wrap on @ and @@.

## Error because __wrap_foo@v1 is not defined.
## Note: GNU ld errors "no symbol version section for versioned symbol `__wrap_foo@v1'".
# RUN: not ld.lld -shared --soname=t --version-script=%t/ver --wrap=foo@v1 %t/ref.o %t/ref1.o %t/def1.o %t/wrap.o \
# RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=W3

# W3:      error: undefined symbol: __wrap_foo@v1
# W3-NEXT: >>> referenced by {{.*}}ref1.o:(.text+0x1)
# W3-NEXT: >>> did you mean: __wrap_foo{{$}}
# W3-NEXT: >>> defined in: {{.*}}wrap.o

## foo@v1 is correctly wrapped.
# RUN: ld.lld -shared --soname=t --version-script=%t/ver --wrap=foo@v1 %t/ref.o %t/ref1.o %t/def1.o %t/wrap1.o -o %t.w4
# RUN: llvm-readobj -r %t.w4 | FileCheck %s --check-prefix=W4REL
# RUN: llvm-objdump -d --no-show-raw-insn %t.w4 | FileCheck %s --check-prefix=W4DIS

# W4REL:      .rela.plt {
# W4REL-NEXT:   R_X86_64_JUMP_SLOT foo@@v1 0x0
# W4REL-NEXT:   R_X86_64_JUMP_SLOT __wrap_foo@v1 0x0
# W4REL-NEXT: }

# W4DIS-LABEL: <.text>:
# W4DIS-NEXT:    callq {{.*}} <foo@plt>
# W4DIS-COUNT-3: int3
# W4DIS-NEXT:    callq {{.*}} <__wrap_foo@plt>

## Note: GNU ld errors "no symbol version section for versioned symbol `__wrap_foo@@v1'".
# RUN: ld.lld -shared --soname=t --version-script=%t/ver --wrap=foo@@v1 %t/ref.o %t/ref1.o %t/def1.o %t/wrap.o -o %t.w5
# RUN: llvm-readobj -r %t.w5 | FileCheck %s --check-prefix=W5REL
# RUN: llvm-objdump -d --no-show-raw-insn %t.w5 | FileCheck %s --check-prefix=W5DIS

# W5REL:      .rela.plt {
# W5REL-NEXT:   R_X86_64_JUMP_SLOT foo@@v1 0x0
# W5REL-NEXT: }

# W5DIS-LABEL: <.text>:
# W5DIS-NEXT:    callq 0x1350 <foo@plt>
# W5DIS-COUNT-3: int3
# W5DIS-NEXT:    callq 0x1350 <foo@plt>

#--- ver
v1 {};
v2 {};

#--- ref.s
call foo

#--- ref1.s
.symver foo, foo@@@v1
call foo

#--- ref1p.s
.protected foo
.symver foo, foo@@@v1
call foo

#--- ref1w.s
.weak foo
.symver foo, foo@@@v1
call foo

#--- def1.s
.globl foo
.symver foo, foo@@@v1
foo:

#--- def1w.s
.weak foo
.symver foo, foo@@@v1
foo:

#--- hid1.s
.globl foo_v1
.symver foo_v1, foo@v1
foo_v1:
  ret

#--- hid1w.s
.weak foo_v1
.symver foo_v1, foo@v1
foo_v1:
  ret

#--- def2.s
.globl foo
.symver foo, foo@@@v2
foo:
  ret

#--- wrap.s
.globl __wrap_foo
__wrap_foo:
  hlt

#--- wrap1.s
.globl __wrap_foo_v1
.symver __wrap_foo_v1, __wrap_foo@v1
__wrap_foo_v1:
  int3
