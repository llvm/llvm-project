# Large data sections

When linking very large binaries, lld may report relocation overflows like

```
relocation R_X86_64_PC32 out of range: 2158227201 is not in [-2147483648, 2147483647]
```

This happens when running into architectural limitations. For example, in x86-64
PIC code, a reference to a static global variable is typically done with a
`R_X86_64_PC32` relocation, which is a 32-bit signed offset from the PC. That
means if the global variable is laid out further than 2GB (2^31 bytes) from the
instruction referencing it, we run into a relocation overflow.

lld normally lays out sections as follows:

```{image} section_layout.png
```

The largest relocation pressure is usually from `.text` to the beginning of
`.rodata` or `.text` to the end of `.bss`.

Some code models offer a tradeoff between relocation pressure and performance.
For example, x86-64's medium code model splits global variables into small and
large globals depending on if their size is over a certain threshold. Large
globals are placed further away from text and we use 64-bit references to refer
to them.

Large globals are placed in separate sections from small globals, and those
sections have a "large" section flag, e.g. `SHF_X86_64_LARGE` for x86-64.
The linker places large sections after all small sections, so that they do not affect the distance between small sections, or between small sections and the ELF header.
The large versions of `.text`, `.rodata`, `.data`, and `.bss` are `.ltext`, `.lrodata`, `.ldata`, and `.lbss`, and they are laid out as follows:

```
.rodata
.text
.data
.bss
.ltext
.lrodata
.ldata
.lbss
```

`.ltext` is the first large section, because small code may reference large code with 32-bit relocations, e.g. when a prebuilt object calls a function compiled with the large code model.
`.lbss`, which can be very large as it takes no space in the file, is the last one.

For non-PIC code, 32-bit absolute relocations typically assume that symbols are in the lower 2GB of the address space.
This layout does not let large sections push small symbols out of that range.

`-z nolrodata-after-bss` places `.ltext` and `.lrodata` before `.rodata` instead, which may need fewer `PT_LOAD` segments:

```
.ltext
.lrodata
.rodata
.text
.data
.bss
.ldata
.lbss
```

With this layout, large `.ltext` and `.lrodata` may cause relocation overflows for references from `.text` to the ELF header, and may push small symbols out of the lower 2GB of the address space.
