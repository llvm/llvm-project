/*===----------------------------------------------------------------------===*
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*
 *
 * \file
 * Bare-metal LLVM libc test linker script.
 *
 *===----------------------------------------------------------------------===*/

ASSERT(DEFINED(__boot_flash), "__boot_flash must be defined");
ASSERT(DEFINED(__boot_flash_size), "__boot_flash_size must be defined");
ASSERT(DEFINED(__flash), "__flash must be defined");
ASSERT(DEFINED(__flash_size), "__flash_size must be defined");
ASSERT(DEFINED(__ram), "__ram must be defined");
ASSERT(DEFINED(__ram_size), "__ram_size must be defined");

MEMORY {
  boot_flash (rx) : ORIGIN = __boot_flash, LENGTH = __boot_flash_size
  flash      (rx) : ORIGIN = __flash,      LENGTH = __flash_size
  ram       (rwx) : ORIGIN = __ram,        LENGTH = __ram_size
}

SECTIONS {
  PROVIDE(__stack = ORIGIN(ram) + LENGTH(ram));

  .boot_flash : {
    KEEP(*(.text.init.enter))
    KEEP(*(.vectors))
    KEEP(*(.vectors.*))
    KEEP(*(.data.init.enter))
  } > boot_flash

  .text : ALIGN(8) {
    KEEP(*(.text.startup .text.startup.*))
    *(.text .text.* .gnu.linkonce.t.*)
  } > flash

  .preinit_array : ALIGN(8) {
    KEEP(*(.preinit_array))
    KEEP(*(.preinit_array.*))
  } > flash

  .init_array : ALIGN(8) {
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP(*(.init_array))
  } > flash

  .fini_array : ALIGN(8) {
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP(*(.fini_array))
  } > flash

  .rodata : ALIGN(8) {
    *(.rodata .rodata.* .gnu.linkonce.r.*)
    *(.srodata .srodata.*)
    *(.rdata)
  } > flash

  .eh_frame_hdr : ALIGN(4) {
    *(.eh_frame_hdr)
    *(.eh_frame_entry .eh_frame_entry.*)
  } > flash

  .eh_frame : ALIGN(4) {
    KEEP(*(.eh_frame))
    KEEP(*(.eh_frame.*))
  } > flash

  .ARM.extab : ALIGN(4) {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > flash

  .ARM.exidx : ALIGN(4) {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > flash

  .data : ALIGN(8) {
    PROVIDE(__data_start = .);
    *(.data .data.* .gnu.linkonce.d.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    . = ALIGN(8);
  } > ram AT> flash
  PROVIDE(__data_source = LOADADDR(.data));
  PROVIDE(__data_size = SIZEOF(.data));

  .bss (NOLOAD) : ALIGN(8) {
    PROVIDE(__bss_start = .);
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.* .gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(8);
    PROVIDE(__bss_size = . - __bss_start);
  } > ram

  .noinit (NOLOAD) : ALIGN(8) {
    *(.noinit .noinit.*)
  } > ram

  PROVIDE(_end = .);

  PROVIDE(__llvm_libc_heap_limit =
              __stack - (DEFINED(__stack_size) ? __stack_size : 0));

  .heap (NOLOAD) : {
    . += (DEFINED(__heap_size_min) ? __heap_size_min : 0);
  } > ram

  .stack (NOLOAD) : {
    . += (DEFINED(__stack_size) ? __stack_size : 0);
  } > ram

  /DISCARD/ : {
    *(.note .note.*)
    *(.comment)
  }
}
