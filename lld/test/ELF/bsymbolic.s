# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld -shared %t/a.o %t/b.o -o %t0.so
# RUN: llvm-readobj -r %t0.so | FileCheck %s --check-prefix=REL_DEF
# RUN: llvm-objdump -d %t0.so | FileCheck %s --check-prefix=ASM_DEF

## -Bsymbolic-functions makes all STT_FUNC definitions non-preemptible.
# RUN: ld.lld -shared -Bsymbolic-functions %t/a.o %t/b.o -o %t1.so
# RUN: llvm-readobj -r %t1.so | FileCheck %s --check-prefix=REL_FUN
# RUN: llvm-objdump -d %t1.so | FileCheck %s --check-prefix=ASM_FUN

## -Bsymbolic makes all definitions non-preemptible.
# RUN: ld.lld -shared -Bsymbolic %t/a.o %t/b.o -o %t2.so
# RUN: llvm-readobj -r %t2.so | FileCheck %s --check-prefix=REL_ALL
# RUN: llvm-objdump -d %t2.so | FileCheck %s --check-prefix=ASM_ALL

# RUN: ld.lld -shared -Bsymbolic-functions -Bsymbolic %t/a.o %t/b.o -o %t.so
# RUN: cmp %t.so %t2.so
# RUN: ld.lld -shared -Bsymbolic -Bsymbolic-functions %t/a.o %t/b.o -o %t.so
# RUN: cmp %t.so %t1.so
# RUN: ld.lld -shared -Bno-symbolic -Bsymbolic %t/a.o %t/b.o -o %t.so
# RUN: cmp %t.so %t2.so

## -Bno-symbolic can cancel previously specified -Bsymbolic and -Bsymbolic-functions.
# RUN: ld.lld -shared -Bsymbolic -Bno-symbolic %t/a.o %t/b.o -o %t.so
# RUN: cmp %t.so %t0.so
# RUN: ld.lld -shared -Bsymbolic-functions -Bno-symbolic %t/a.o %t/b.o -o %t.so
# RUN: cmp %t.so %t0.so

# REL_DEF:      .rela.dyn {
# REL_DEF-NEXT:   R_X86_64_RELATIVE -
# REL_DEF-NEXT:   R_X86_64_RELATIVE -
# REL_DEF-NEXT:   R_X86_64_64 data_default
# REL_DEF-NEXT: }
# REL_DEF-NEXT: .rela.plt {
# REL_DEF-NEXT:   R_X86_64_JUMP_SLOT default
# REL_DEF-NEXT:   R_X86_64_JUMP_SLOT ext_default
# REL_DEF-NEXT:   R_X86_64_JUMP_SLOT notype_default
# REL_DEF-NEXT:   R_X86_64_JUMP_SLOT undef
# REL_DEF-NEXT: }

# ASM_DEF:      <_start>:
# ASM_DEF-NEXT:   callq {{.*}} <default@plt>
# ASM_DEF-NEXT:   callq {{.*}} <protected>
# ASM_DEF-NEXT:   callq {{.*}} <hidden>
# ASM_DEF-NEXT:   callq {{.*}} <ext_default@plt>
# ASM_DEF-NEXT:   callq {{.*}} <notype_default@plt>
# ASM_DEF-NEXT:   callq {{.*}} <undef@plt>

# REL_FUN:      .rela.dyn {
# REL_FUN-NEXT:   R_X86_64_RELATIVE -
# REL_FUN-NEXT:   R_X86_64_RELATIVE -
# REL_FUN-NEXT:   R_X86_64_64 data_default
# REL_FUN-NEXT: }
# REL_FUN-NEXT: .rela.plt {
# REL_FUN-NEXT:   R_X86_64_JUMP_SLOT notype_default
# REL_FUN-NEXT:   R_X86_64_JUMP_SLOT undef
# REL_FUN-NEXT: }

# ASM_FUN:      <_start>:
# ASM_FUN-NEXT:   callq {{.*}} <default>
# ASM_FUN-NEXT:   callq {{.*}} <protected>
# ASM_FUN-NEXT:   callq {{.*}} <hidden>
# ASM_FUN-NEXT:   callq {{.*}} <ext_default>
# ASM_FUN-NEXT:   callq {{.*}} <notype_default@plt>
# ASM_FUN-NEXT:   callq {{.*}} <undef@plt>

# REL_ALL:      .rela.dyn {
# REL_ALL-NEXT:   R_X86_64_RELATIVE -
# REL_ALL-NEXT:   R_X86_64_RELATIVE -
# REL_ALL-NEXT:   R_X86_64_RELATIVE -
# REL_ALL-NEXT: }
# REL_ALL-NEXT: .rela.plt {
# REL_ALL-NEXT:   R_X86_64_JUMP_SLOT undef
# REL_ALL-NEXT: }

# ASM_ALL:      <_start>:
# ASM_ALL-NEXT:   callq {{.*}} <default>
# ASM_ALL-NEXT:   callq {{.*}} <protected>
# ASM_ALL-NEXT:   callq {{.*}} <hidden>
# ASM_ALL-NEXT:   callq {{.*}} <ext_default>
# ASM_ALL-NEXT:   callq {{.*}} <notype_default>
# ASM_ALL-NEXT:   callq {{.*}} <undef@plt>

#--- a.s
.globl default, protected, hidden, notype_default
.protected protected
.hidden hidden
.type default, @function
.type protected, @function
.type hidden, @function
default: nop
protected: nop
hidden: nop
notype_default: nop

.globl _start
_start:
  callq default@PLT
  callq protected@PLT
  callq hidden@PLT

  callq ext_default@PLT

  callq notype_default@PLT

  callq undef@PLT

.data
  .quad data_default
  .quad data_protected
  .quad data_hidden

.globl data_default, data_protected, data_hidden
.protected data_protected
.hidden data_hidden
.type data_default, @object
.type data_protected, @object
.type data_hidden, @object
data_default: .byte 0
data_protected: .byte 0
data_hidden: .byte 0

#--- b.s
.globl ext_default
.type ext_default,@function
ext_default:
  nop
