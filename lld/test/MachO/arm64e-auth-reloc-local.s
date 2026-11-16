# REQUIRES: aarch64
## The assembler emits an authenticated pointer to a local symbol as an extern
## relocation, which keeps its signing schema. ld64 rejects the non-extern
## (section-relative) form, and so does lld.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/local.o %t/local.s
# RUN: llvm-objdump --macho -r %t/local.o | FileCheck %s --check-prefix=OBJ
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib %t/local.o \
# RUN:   -o %t/local.dylib
# RUN: llvm-objdump --macho --syms %t/local.dylib > %t/out
# RUN: llvm-objdump --macho --dyld-info %t/local.dylib >> %t/out
# RUN: FileCheck %s < %t/out

# RUN: yaml2obj %t/nonextern.yaml -o %t/nonextern.o
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   %t/nonextern.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# OBJ: 00000000 False ?( 3) True 11 (?) False _target

# CHECK:       [[#%x,TARGET:]] l O __DATA_CONST,__const _target
## An authenticated rebase of _target with key DA, diversity 0x1234 and
## address diversity.
# CHECK-LABEL: dyld information:
# CHECK:       __DATA __data 0x{{[0-9A-F]+}} 0x{{[0-9A-F]+}} rebase 0x1234 true DA 0x[[#%X,TARGET]]

# ERR: error: AUTHENTICATED_POINTER relocation must be extern at offset 0 of __DATA,__data in {{.*}}nonextern.o

#--- local.s
.section __DATA,__const
.p2align 3
_target:
  .quad 0

.section __DATA,__data
.p2align 3
.globl _ptr
_ptr:
  .quad _target@AUTH(da,0x1234,addr)

## local.s with the relocation turned into a section-relative one.
#--- nonextern.yaml
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      0x80000002
  filetype:        0x1
  ncmds:           2
  sizeofcmds:      256
  flags:           0x0
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         ''
    vmaddr:          0
    vmsize:          16
    fileoff:         288
    filesize:        16
    maxprot:         7
    initprot:        7
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __const
        segname:         __DATA
        addr:            0x0
        size:            8
        offset:          0x120
        align:           3
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '0000000000000000'
      - sectname:        __data
        segname:         __DATA
        addr:            0x8
        size:            8
        offset:          0x128
        align:           3
        reloff:          0x130
        nreloc:          1
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '0000000034120580'
        relocations:
          - address:         0x0
            symbolnum:       1
            pcrel:           false
            length:          3
            extern:          false
            type:            11
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          312
    nsyms:           1
    stroff:          328
    strsize:         6
LinkEditData:
  NameList:
    - n_strx:          1
      n_type:          0xF
      n_sect:          2
      n_desc:          0
      n_value:         8
  StringTable:
    - ''
    - _ptr
...
