/* Reserve space for BOLT to rewrite .eh_frame in place. */
SECTIONS {
  .eh_frame : {
    *(.eh_frame)
    . += 0x80;
  }
} INSERT AFTER .rodata;
