// REQUIRES: arm-registered-target

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang --target=armv7-unknown-none-eabi -march=armv7a \
// RUN:   -fPIC -mno-pic-data-is-text-relative -ffreestanding -fno-builtin \
// RUN:   -c %t/main.c -o %t/main.o
// RUN: %clang --target=armv7-unknown-none-eabi -march=armv7a \
// RUN:   -fPIC -mno-pic-data-is-text-relative -x assembler-with-cpp \
// RUN:   -c %t/start.S -o %t/start.o
// RUN: llvm-readobj --relocations %t/main.o | FileCheck %s --check-prefix=MAIN-REL
// RUN: llvm-readobj --relocations %t/start.o | FileCheck %s --check-prefix=STARTUP-REL
// RUN: ld.lld -T %t/proof.ld %t/start.o %t/main.o -o %t/proof.elf
// RUN: llvm-readobj --sections --symbols %t/proof.elf | FileCheck %s --check-prefix=LINK

// MAIN-REL-DAG: R_ARM_GOT_BREL global_counter
// MAIN-REL-DAG: R_ARM_GOT_BREL global_delta

// STARTUP-REL: R_ARM_REL32 __pic_static_base

// LINK: Name: .got
// LINK: Name: __pic_static_base
// LINK: Name: global_counter

//--- main.c
volatile int global_counter = 41;
volatile const int global_delta = 1;

int main(void) {
  global_counter += global_delta;
  return global_counter;
}

//--- start.S
#ifndef __ARM_PIC_DATA_STATIC_BASE
#error "static-base PIC data macro missing"
#endif

  .syntax unified
  .arm
  .globl _start
  .type _start, %function
_start:
  adr r9, .Lpic_static_base
  ldr r12, [r9]
  add r9, r9, r12
  bl main
.Lhang:
  b .Lhang
  .balign 4
.Lpic_static_base:
  .word __pic_static_base - .
  .size _start, . - _start

//--- proof.ld
ENTRY(_start)

SECTIONS {
  . = 0x10000;

  .text : {
    *(.text*)
    *(.rodata*)
  }

  .got : {
    PROVIDE_HIDDEN(__pic_static_base = .);
    PROVIDE_HIDDEN(_GLOBAL_OFFSET_TABLE_ = .);
    *(.got)
    *(.got.*)
  }

  .data : { *(.data*) }
  .bss : { *(.bss*) *(COMMON) }
}
