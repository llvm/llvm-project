# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=NORELOC %s
# RUN: llvm-readelf -x .data %t | FileCheck --check-prefix=DATA %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=RELOC2 %s
# RUN: llvm-readelf -x .data %t.so | FileCheck --check-prefix=DATA2 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=DISASM2 %s

# NORELOC:      Relocations [
# NORELOC-NEXT: ]

# DATA:      section '.data':
# DATA-NEXT: 0x002031ac 00000000 00000000 00000000 00000000
# DATA-NEXT: 0x002031bc 00000000 00000000 00001900 00000000
# DATA-NEXT: 0x002031cc 00001b00 00000000 00001900 00000000
# DATA-NEXT: 0x002031dc 00001b00 00000000 0000

# DISASM:      _start:
# DISASM-NEXT:   movl 25, %eax
# DISASM-NEXT:   movl 27, %eax
# DISASM-NEXT:   movl 25, %eax
# DISASM-NEXT:   movl 27, %eax

# RELOC2:      Relocations [
# RELOC2-NEXT: Section ({{.*}}) .rela.dyn {
# RELOC2-NEXT:    0x2333 R_X86_64_SIZE32 foo 0xFFFFFFFFFFFFFFFF
# RELOC2-NEXT:    0x233A R_X86_64_SIZE32 foo 0x1
# RELOC2-NEXT:    0x440A R_X86_64_SIZE64 foo 0xFFFFFFFFFFFFFFFF
# RELOC2-NEXT:    0x4412 R_X86_64_SIZE64 foo 0x1
# RELOC2-NEXT:  }
# RELOC2-NEXT: ]

# DATA2:      section '.data':
# DATA2-NEXT: 00000000 00000000 00000000 00000000
# DATA2-NEXT: 00000000 00000000 00000000 00000000
# DATA2-NEXT: 00000000 00000000 00001900 00000000
# DATA2-NEXT: 00001b00 00000000 0000

# DISASM2:      _start:
# DISASM2-NEXT:   movl 0, %eax
# DISASM2-NEXT:   movl 0, %eax
# DISASM2-NEXT:   movl 25, %eax
# DISASM2-NEXT:   movl 27, %eax

.data
.global foo, foohidden
.hidden foohidden
.type foo,%object
.size foo,26
.type foohidden,%object
.size foohidden,26
foo:
foohidden:
.zero 26

  // R_X86_64_SIZE64:
  .quad foo@SIZE-1
  .quad foo@SIZE+1
  .quad foohidden@SIZE-1
  .quad foohidden@SIZE+1

.section test,"axw"
.globl _start
_start:
  // R_X86_64_SIZE32:
  movl foo@SIZE-1,%eax
  movl foo@SIZE+1,%eax
  movl foohidden@SIZE-1,%eax
  movl foohidden@SIZE+1,%eax
