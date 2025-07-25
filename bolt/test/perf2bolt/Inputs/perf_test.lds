SECTIONS {
  . = 0x400000 + SIZEOF_HEADERS;
  .interp : { *(.interp) }
  .note.gnu.build-id : { *(.note.gnu.build-id) }
  .dynsym         : { *(.dynsym) }
  . = 0x801000;
  .text : { *(.text*) }
  . = 0x803000;
  .fini_array : { *(.fini_array) }
  . = 0x805000;
  .data : { *(.data) }
}
