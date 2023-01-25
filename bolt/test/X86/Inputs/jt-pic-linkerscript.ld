# Linker script used by jump-table-pic-conflict.s test.
# .rodata needs to appear before .text

SECTIONS
{
  . = 0x201120;
  .rodata : { *(.rodata) }
  .eh_frame : { *(.eh_frame) }
  .text : { *(.text) }
}
