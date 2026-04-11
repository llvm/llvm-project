## Test input for stabs-icf.s: local (non-global) symbols that will be ICF'd.
## _local_baz and _local_baz2 have identical bodies to _local_bar and will be
## folded. Since they are local symbols, duplicate nlist entries at the same
## address cause atos to return "<deduplicated_symbol>".

.text
.globl _main

.subsections_via_symbols

_local_bar:
  ret

_local_baz:
  ret

_local_baz2:
  ret

_main:
Lfunc_begin0:
  call _local_bar
  call _local_baz
  call _local_baz2
  ret
Lfunc_end0:

.section  __DWARF,__debug_str,regular,debug
  .asciz  "test.cpp"
  .asciz  "/tmp"
.section  __DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
  .byte  1
  .byte  17
  .byte  1
  .byte  3
  .byte  14
  .byte  27
  .byte  14
  .byte  17
  .byte  1
  .byte  18
  .byte  6
  .byte  0
  .byte  0
  .byte  0
.section  __DWARF,__debug_info,regular,debug
.set Lset0, Ldebug_info_end0-Ldebug_info_start0
  .long  Lset0
Ldebug_info_start0:
  .short  4
.set Lset1, Lsection_abbrev-Lsection_abbrev
  .long  Lset1
  .byte  8
  .byte  1
  .long  0
  .long  9
  .quad  Lfunc_begin0
.set Lset3, Lfunc_end0-Lfunc_begin0
  .long  Lset3
  .byte  0
Ldebug_info_end0:
