SECTIONS {
    . = 0;
    . = ALIGN(0x400000);
    .text : {
        *(foo_section)
        . += 0x7BFFFFC;
        *(main_section)
        ASSERT(foo == 0x400000, "Error: foo address is not 0x400000.");
        ASSERT(_start == 0x8000000, "Error: _start address is not 0x8000000.");
    }
}
