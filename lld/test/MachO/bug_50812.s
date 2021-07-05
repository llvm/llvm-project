# REQUIRES: x86
## FIXME: This yaml is from an object file produced with 'ld -r'
##        Replace this with "normal" .s test format once lld supports `-r`

# RUN: yaml2obj %s -o %t.o
# RUN: %lld -lSystem -platform_version macos 11.3 11.0 -arch x86_64 %t.o -o %t
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x01000007
  cpusubtype:      0x00000003
  filetype:        0x00000001
  ncmds:           2
  sizeofcmds:      384
  flags:           0x00002000
  reserved:        0x00000000
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         312
    segname:         ''
    vmaddr:          0
    vmsize:          120
    fileoff:         448
    filesize:        120
    maxprot:         7
    initprot:        7
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0000000000000000
        size:            18
        offset:          0x000001C0
        align:           4
        reloff:          0x00000000
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
        content:         554889E5C745FC00000000B8010000005DC3
      - sectname:        __eh_frame
        segname:         __TEXT
        addr:            0x0000000000000018
        size:            64
        offset:          0x000001D8
        align:           3
        reloff:          0x00000238
        nreloc:          4
        flags:           0x00000000
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
        content:         1400000000000000017A520001781001100C0708900100002400000004000000F8FFFFFFFFFFFFFF120000000000000000410E108602430D0600000000000000
        relocations:
          - address:         0x0000001C
            symbolnum:       0
            pcrel:           false
            length:          2
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x0000001C
            symbolnum:       1
            pcrel:           false
            length:          2
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x00000020
            symbolnum:       1
            pcrel:           false
            length:          3
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x00000020
            symbolnum:       10
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          608
    nsyms:           11
    stroff:          784
    strsize:         72
LinkEditData:
  NameList:
    - n_strx:          8      ## N_STAB sym (in got)
      n_type:          0x0E
      n_sect:          2
      n_desc:          0
      n_value:         24
    - n_strx:          18
      n_type:          0x0E
      n_sect:          2
      n_desc:          0
      n_value:         48
    - n_strx:          1
      n_type:          0x4E
      n_sect:          1
      n_desc:          0
      n_value:         18
    - n_strx:          2          ## _main
      n_type:          0x0F
      n_sect:          1
      n_desc:          0
      n_value:         0
  StringTable:
    - ' '
    - _main
    - EH_Frame1
    - func.eh
    - '/Users/vyng/'
    - test.cc
    - '/Users/vyng/test.o'
    - _main
...
