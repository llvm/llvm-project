## When changing the assembly input, uncomment these lines to re-generate the
## YAML.
# COM: llvm-mc --emit-dwarf-unwind=always -filetype=obj -triple=x86_64-apple-macos10.15 %s -o %t.o
# COM: ld -r %t.o -o %t-r.o
# COM: obj2yaml %t-r.o > %S/Inputs/double-unwind-info.yaml

# RUN: yaml2obj %S/Inputs/double-unwind-info.yaml > %t-r.o
# RUN: %lld -dylib -lSystem %t-r.o -o /dev/null

.text
## eh_frame function address relocations are only emitted if the function isn't
## at address 0x0.
_spacer:
  nop

## Check that we perform unwind info registration correctly when there are
## multiple symbols at the same address. This would previously hit an assertion
## error (PR56570).
_foo:
ltmp1:
  .cfi_startproc
  .cfi_def_cfa_offset 8
  nop
  .cfi_endproc

.subsections_via_symbols
