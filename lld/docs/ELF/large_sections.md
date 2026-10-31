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
sections have a "large" section flag, e.g. `SHF_X86_64_LARGE` for x86-64. The
linker places large sections after the small sections, making sure they
do not affect the distance of small globals to text. The large versions
of `.rodata`, `.bss`, and `.data` are `.lrodata`, `.lbss`, and
`.ldata`, and they are laid out as follows:

```{image} large_section_layout_nopic.png
```

Large code is placed in `.ltext`, which is laid out after `.lbss` and before
`.lrodata`.

`.lbss` is right after `.bss` so that they are merged together and we
minimize the number of segments with `p_memsz > p_filesz`.

For non-PIC code with absolute relocations, 32-bit signed relocations
typically assume that symbols are in the lower 2GB of the address space. So for
non-PIC code, large sections should be placed after all small sections to avoid
`.lrodata` pushing small symbols out of the lower 2GB of the address space.
This layout also keeps `.text` close to the ELF headers, allowing small-model
PIC code to reference `__ehdr_start` with a 32-bit PC-relative relocation.

`-z lrodata-after-bss` is the default. `-z nolrodata-after-bss` places
`.lrodata` before `.text`:

```{image} large_section_layout_pic.png
```

`.ltext` remains after `.bss` regardless of this option.
