# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11 %t/signed.s -o %t/signed.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11 %t/lib.s -o %t/lib.o

## Undefined symbols are looked up dynamically regardless of -undefined, and
## -static, which clang passes for kexts, is accepted without a warning.
# RUN: %no-lsystem-lld -kext -static -undefined error -o %t/main.kext %t/main.o
# RUN: llvm-objdump --macho -r --private-headers %t/main.kext | \
# RUN:   FileCheck %s --implicit-check-not="cmd LC_"
# RUN: llvm-objdump --macho -d --no-show-raw-insn %t/main.kext | \
# RUN:   FileCheck %s --check-prefix=DISASM
# RUN: llvm-objdump --macho -s --section=__DATA,__data %t/main.kext | \
# RUN:   FileCheck %s --check-prefix=DATA
# RUN: llvm-nm -m %t/main.kext | FileCheck %s --check-prefix=SYMS

## Branches and pointers to undefined symbols, including their GOT slots, are
## bound through external relocations. Local relocations name the section that
## they point into, and references to weak definitions are not bound.
# CHECK:      External relocation information 3 entries
# CHECK-NEXT: address  pcrel length extern type    scattered symbolnum/value
# CHECK-NEXT: [[#%.8x,GOT:]] False quad   True   UNSIGND False     _ext_data
# CHECK-NEXT: [[#%.8x,CALL:]] True  long   True   BRANCH  False     _ext_func
# CHECK-NEXT: [[#%.8x,EXTPTR:]] False quad   True   UNSIGND False     _ext_data
# CHECK-NEXT: Local relocation information 3 entries
# CHECK-NEXT: address  pcrel length extern type    scattered symbolnum/value
# CHECK-NEXT: [[#%.8x,EXTPTR-16]] False quad   False  UNSIGND False     1 (__TEXT,__text)
# CHECK-NEXT: [[#%.8x,EXTPTR-8]] False quad   False  UNSIGND False     2 (__TEXT,__cstring)
# CHECK-NEXT: [[#%.8x,EXTPTR+8]] False quad   False  UNSIGND False     4 (__DATA,__data)

## Like ld64, there is no __DATA_CONST segment, and no LC_BUILD_VERSION,
## LC_FUNCTION_STARTS, or dyld info. The GOT is a regular section.
# CHECK:      filetype ncmds sizeofcmds flags
# CHECK-NEXT: KEXTBUNDLE 6 {{[0-9]+}} NOUNDEFS DYLDLINK TWOLEVEL{{$}}
# CHECK:      cmd LC_SEGMENT_64
# CHECK-NEXT: cmdsize
# CHECK-NEXT: segname __TEXT
# CHECK-NEXT: vmaddr 0x0000000000000000
# CHECK:      sectname __text
# CHECK-NEXT: segname __TEXT
# CHECK-NEXT: addr 0x[[#%.16x,CALL-1]]
# CHECK:      sectname __cstring
# CHECK:      cmd LC_SEGMENT_64
# CHECK-NEXT: cmdsize
# CHECK-NEXT: segname __DATA
# CHECK:      sectname __got
# CHECK-NEXT: segname __DATA
# CHECK-NEXT: addr 0x[[#%.16x,GOT]]
# CHECK:      type S_REGULAR
# CHECK-NEXT: attributes (none)
# CHECK-NEXT: reserved1 0
# CHECK:      sectname __data
# CHECK-NEXT: segname __DATA
# CHECK-NEXT: addr 0x[[#%.16x,EXTPTR-16]]
# CHECK:      cmd LC_SEGMENT_64
# CHECK-NEXT: cmdsize
# CHECK-NEXT: segname __LINKEDIT
# CHECK:      cmd LC_SYMTAB
# CHECK:      cmd LC_DYSYMTAB
# CHECK:      nindirectsyms 1
# CHECK-NEXT: extreloff
# CHECK-NEXT: nextrel 3
# CHECK-NEXT: locreloff
# CHECK-NEXT: nlocrel 3
# CHECK:      cmd LC_UUID

## The load from the GOT of a weak definition is relaxed, and calls to it are
## direct.
# DISASM:      _start_fn:
# DISASM-NEXT: callq _ext_func
# DISASM-NEXT: movq 0x{{[0-9a-f]+}}(%rip), %rax
# DISASM-NEXT: leaq _weak_data(%rip), %rax
# DISASM-NEXT: callq _weak_fn

## The addend of an external relocation is left in place.
# DATA:      Contents of (__DATA,__data) section
# DATA-NEXT: {{^[0-9a-f]+}}
# DATA-NEXT: {{^[0-9a-f]+}} 08 00 00 00 00 00 00 00

## ld64 defines no header symbol for kexts.
# SYMS-NOT: __mh_
# SYMS:     (undefined) external _ext_data (dynamically looked up)
# SYMS:     (undefined) external _ext_func (dynamically looked up)

## Like ld64, kexts are never linked against dylibs.
# RUN: %no-lsystem-lld -dylib -o %t/libfoo.dylib %t/lib.o
# RUN: llvm-ar rcs %t/libbar.a %t/lib.o
# RUN: not %no-lsystem-lld -kext -o /dev/null %t/main.o -L%t -lfoo 2>&1 | \
# RUN:   FileCheck %s --check-prefix=NOLIB
# RUN: %no-lsystem-lld -kext -o /dev/null %t/main.o -L%t -lbar
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 -kext \
# RUN:   -o /dev/null %t/main.o %t/libfoo.dylib 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DYLIB -DFILE=%t/libfoo.dylib
# NOLIB: error: library not found for -lfoo
# DYLIB: warning: ignoring unexpected dylib '[[FILE]]'

# RUN: not %no-lsystem-lld -kext -o /dev/null %t/signed.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=SIGNED
# SIGNED: error: {{.*}}signed.o:(symbol _f+0x3): SIGNED relocation to _ext cannot be bound in a kext; access it through the GOT instead

# RUN: not %no-lsystem-lld -kext -fixup_chains -o /dev/null %t/main.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHAINED
# CHAINED: error: -fixup_chains is incompatible with -kext

# RUN: not %no-lsystem-lld -kext -arch arm64 -o /dev/null %t/main.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ARM64
# ARM64: error: -kext is only supported for x86_64 targets

#--- main.s
.text
.globl _start_fn
_start_fn:
  callq _ext_func
  movq _ext_data@GOTPCREL(%rip), %rax
  movq _weak_data@GOTPCREL(%rip), %rax
  callq _weak_fn
  retq

.globl _weak_fn
.weak_definition _weak_fn
_weak_fn:
  retq

.cstring
Lstr:
  .asciz "kext"

.data
_local_ptr:
  .quad _start_fn
_section_ptr:
  .quad Lstr
_ext_ptr:
  .quad _ext_data + 8
_weak_ptr:
  .quad _weak_data

.globl _weak_data
.weak_definition _weak_data
_weak_data:
  .quad 0

.subsections_via_symbols

#--- signed.s
.text
.globl _f
_f:
  leaq _ext(%rip), %rax
  retq

#--- lib.s
.globl _ext_func
_ext_func:
  retq
