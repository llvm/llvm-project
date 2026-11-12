# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/regular.s -o %t/regular.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/tbss.s -o %t/tbss.o

# RUN: %lld -lSystem -no_pie -o %t/regular-no-pie %t/regular.o
# RUN: llvm-otool -hv %t/regular-no-pie | FileCheck %s --check-prefix=HEADER
# RUN: llvm-objdump -d --bind --rebase %t/regular-no-pie | FileCheck %s --check-prefixes=REG,LINKEDIT
# RUN: llvm-objdump --macho --section=__DATA,__thread_vars %t/regular-no-pie | \
# RUN:   FileCheck %s --check-prefix=REG-TLVP

# RUN: %lld -lSystem %t/regular.o -o %t/regular-pie
# RUN: llvm-otool -hv %t/regular-pie | FileCheck %s --check-prefix=HEADER
# RUN: llvm-objdump -d --bind --rebase %t/regular-pie | FileCheck %s --check-prefixes=REG,LINKEDIT
# RUN: llvm-objdump --macho --section=__DATA,__thread_vars %t/regular-pie | \
# RUN:   FileCheck %s --check-prefix=REG-TLVP

# RUN: %lld -lSystem %t/tbss.o -o %t/tbss -e _f
# RUN: llvm-objdump -d --bind --rebase %t/tbss | FileCheck %s --check-prefixes=TBSS,LINKEDIT
# RUN: llvm-objdump --macho --section=__DATA,__thread_vars %t/tbss | \
# RUN:   FileCheck %s --check-prefix=TBSS-TLVP

# RUN: %lld -lSystem %t/regular.o %t/tbss.o -o %t/regular-and-tbss
# RUN: llvm-objdump -d --bind --rebase %t/regular-and-tbss | FileCheck %s --check-prefixes=REG,TBSS,LINKEDIT
# RUN: llvm-objdump --macho --section=__DATA,__thread_vars %t/regular-and-tbss | \
# RUN:   FileCheck %s --check-prefix=REG-TBSS-TLVP
# RUN: llvm-objdump --section-headers %t/regular-and-tbss | FileCheck %s --check-prefix=SECTIONS

## Check that we always put __thread_bss immediately after __thread_data,
## regardless of the order of the input files.
# RUN: %lld -lSystem %t/tbss.o %t/regular.o -o %t/regular-and-tbss
# RUN: llvm-objdump --section-headers %t/regular-and-tbss | FileCheck %s --check-prefix=SECTIONS

# HEADER: MH_HAS_TLV_DESCRIPTORS

# REG:       <_main>:
# REG-NEXT:  leaq    {{.*}}(%rip), %rax  ## {{.*}} <_foo>
# REG-NEXT:  leaq    {{.*}}(%rip), %rax  ## {{.*}} <_bar>
# REG-NEXT:  retq

# TBSS:       <_f>:
# TBSS-NEXT:  leaq    {{.*}}(%rip), %rax  ## {{.*}} <_baz>
# TBSS-NEXT:  leaq    {{.*}}(%rip), %rax  ## {{.*}} <_qux>
# TBSS-NEXT:  leaq    {{.*}}(%rip), %rax  ## {{.*}} <_hoge>
# TBSS-NEXT:  retq

# REG-TLVP:      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TLVP-NEXT: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TLVP-NEXT: 00 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00

# TBSS-TLVP:      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00

# REG-TBSS-TLVP:      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 28 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
# REG-TBSS-TLVP-NEXT: 30 00 00 00 00 00 00 00

## Make sure we don't emit rebase opcodes for relocations in __thread_vars.
# LINKEDIT:       Rebase table:
# LINKEDIT-NEXT:  segment  section            address     type
# LINKEDIT-EMPTY:
# LINKEDIT-NEXT:  Bind table:
# LINKEDIT:       __DATA  __thread_vars   0x{{[0-9a-f]*}}  pointer 0 libSystem __tlv_bootstrap
# LINKEDIT:       __DATA  __thread_vars   0x{{[0-9a-f]*}}  pointer 0 libSystem __tlv_bootstrap

## Make sure we have an odd number of tlv vars, and that the __thread_vars
## section starts 16-bytes aligned. This is the setup required for __thread_data
## not to be automatically 16-bytes aligned, ensuring the linker does its
## expected job of aligning _hoge$tlv$init.
# SECTIONS:      __thread_vars {{[0-9]+}}8 {{[0-9]+}}0
# SECTIONS:      __thread_data
# SECTIONS:      more_thread_data
# SECTIONS-NEXT: __thread_bss

#--- regular.s
.globl _main
_main:
  mov _foo@TLVP(%rip), %rax
  mov _bar@TLVP(%rip), %rax
  ret

.section __DATA,__thread_data,thread_local_regular
_foo$tlv$init:
  .quad 123

.section __DATA,more_thread_data,thread_local_regular
_bar$tlv$init:
  .quad 123

.section __DATA,__thread_vars,thread_local_variables
.globl  _foo, _bar
_foo:
  .quad  __tlv_bootstrap
  .quad  0
  .quad  _foo$tlv$init
_bar:
  .quad  __tlv_bootstrap
  .quad  0
  .quad  _bar$tlv$init

#--- tbss.s

.globl _f
_f:
  mov _baz@TLVP(%rip), %rax
  mov _qux@TLVP(%rip), %rax
  mov _hoge@TLVP(%rip), %rax
  ret

.tbss _baz$tlv$init, 8, 3
.tbss _qux$tlv$init, 8, 3
.tbss _hoge$tlv$init, 16, 4

.section __DATA,__thread_vars,thread_local_variables
_baz:
  .quad  __tlv_bootstrap
  .quad  0
  .quad  _baz$tlv$init
_qux:
  .quad  __tlv_bootstrap
  .quad  0
  .quad  _qux$tlv$init
_hoge:
  .quad  __tlv_bootstrap
  .quad  0
  .quad  _hoge$tlv$init
