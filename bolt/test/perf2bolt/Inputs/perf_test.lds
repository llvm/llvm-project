SECTIONS {
  . = SIZEOF_HEADERS;
  .interp : { *(.interp) }
  .note.gnu.build-id : { *(.note.gnu.build-id) }
  . = 0x212e8;
  .dynsym         : { *(.dynsym) }
  . = 0x31860;
  .text : { *(.text*) }
  . = 0x41c20;
  .fini_array : { *(.fini_array) }
  . = 0x54e18;
  .data : { *(.data) }
}