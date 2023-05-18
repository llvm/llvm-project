# REQUIRES: x86

# RUN: echo -e ".global variable\n.global DllMainCRTStartup\n.text\nDllMainCRTStartup:\nret\n.data\nvariable:\n.long 42" > %t-lib.s
# RUN: llvm-mc -triple=x86_64-windows-gnu %t-lib.s -filetype=obj -o %t-lib.obj
# RUN: lld-link -out:%t-lib.dll -dll -entry:DllMainCRTStartup %t-lib.obj -lldmingw -implib:%t-lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu %s -defsym listptrs=1 -filetype=obj -o %t.obj
# RUN: lld-link -lldmingw -out:%t.exe -entry:main %t.obj %t-lib.lib -verbose

# RUN: llvm-readobj --coff-imports %t.exe | FileCheck -check-prefix=IMPORTS %s
# RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck --check-prefix=DISASM %s
# RUN: llvm-objdump -s %t.exe | FileCheck --check-prefix=CONTENTS %s

## Check that we can autoimport these variables with pseudo relocs disabled.
# RUN: llvm-mc -triple=x86_64-windows-gnu %s -defsym listptrs=0 -filetype=obj -o %t.noptrs.obj
# RUN: lld-link -lldmingw -runtime-pseudo-reloc:no -out:%t.exe -entry:main %t.noptrs.obj %t-lib.lib

## Check that we can't autoimport them with autoimport disabled.
# RUN: not lld-link -lldmingw -auto-import:no -out:%t.exe -entry:main %t.noptrs.obj %t-lib.lib 2>&1 | FileCheck --check-prefix=NO-AUTOIMPORT %s

# IMPORTS: Import {
# IMPORTS-NEXT: Name: autoimport-refptr.s.tmp-lib.dll
# IMPORTS-NEXT: ImportLookupTableRVA: 0x2050
# IMPORTS-NEXT: ImportAddressTableRVA: 0x2060
# IMPORTS-NEXT: Symbol: variable (0)
# IMPORTS-NEXT: }

# DISASM: Disassembly of section .text:
# DISASM-EMPTY:
# DISASM: <.text>:
# Relative offset at 0x1002 pointing at the IAT at 0x2060
# DISASM: 140001000:      48 8b 05 59 10 00 00    movq    4185(%rip), %rax
# DISASM: 140001007:      8b 00   movl    (%rax), %eax
# Relative offset at 0x100b pointing at the .refptr.localvar stub at
# 0x2000
# DISASM: 140001009:      48 8b 0d f0 0f 00 00    movq    4080(%rip), %rcx
# DISASM: 140001010:      03 01   addl    (%rcx), %eax
# DISASM: 140001012:      c3      retq

# relocs: pointing at an empty list of runtime pseudo relocs.
# localvar: 42
# CONTENTS: Contents of section .data:
# CONTENTS:  140003000 08200040 01000000 08200040 01000000
# CONTENTS:  140003010 2a000000

# NO-AUTOIMPORT: error: undefined symbol: variable

    .global main
    .global localvar
    .text
main:
    movq .refptr.variable(%rip), %rax
    movl (%rax), %eax
    movq .refptr.localvar(%rip), %rcx
    addl (%rcx), %eax
    ret

    .data
.if listptrs==1
relocs:
    .quad __RUNTIME_PSEUDO_RELOC_LIST__
    .quad __RUNTIME_PSEUDO_RELOC_LIST_END__
.endif
localvar:
    .int 42

# Normally the compiler wouldn't emit a stub for a variable that is
# emitted in the same translation unit.
    .section .rdata$.refptr.localvar,"dr",discard,.refptr.localvar
    .global .refptr.localvar
.refptr.localvar:
    .quad localvar

    .section .rdata$.refptr.variable,"dr",discard,.refptr.variable
    .global .refptr.variable
.refptr.variable:
    .quad variable
