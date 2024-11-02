# REQUIRES: x86
## FIXME: This yaml is from an object file produced with 'ld -r':
##
##   echo "int main() {return 1;}" > test.c
##   clang -c -g -o test.o test.c
##   ld -r -o test2.o test.o -no_data_in_code_info
##
## Replace this with "normal" .s test format once lld supports `-r`

# RUN: yaml2obj %s -o %t.o
# RUN: %lld -lSystem -arch x86_64 %t.o -o %t

--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x1000007
  cpusubtype:      0x3
  filetype:        0x1
  ncmds:           3
  sizeofcmds:      288
  flags:           0x2000
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         232
    segname:         ''
    vmaddr:          0
    vmsize:          56
    fileoff:         352
    filesize:        56
    maxprot:         7
    initprot:        7
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0
        size:            18
        offset:          0x160
        align:           4
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         554889E5C745FC00000000B8010000005DC3
      - sectname:        __compact_unwind
        segname:         __LD
        addr:            0x18
        size:            32
        offset:          0x178
        align:           3
        reloff:          0x198
        nreloc:          1
        flags:           0x2000000
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '0000000000000000120000000000000100000000000000000000000000000000'
        relocations:
          - address:         0x0
            symbolnum:       8
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          416
    nsyms:           9
    stroff:          560
    strsize:         48
  - cmd:             LC_BUILD_VERSION
    cmdsize:         32
    platform:        1
    minos:           659200
    sdk:             0
    ntools:          1
    Tools:
      - tool:            3
        version:         46596096
LinkEditData:
  NameList:
    - n_strx:          8
      n_type:          0x64 ## N_SO STAB
      n_sect:          0
      n_desc:          0
      n_value:         0
    - n_strx:          14
      n_type:          0x64 ## N_SO STAB
      n_sect:          0
      n_desc:          0
      n_value:         0
    - n_strx:          21
      n_type:          0x66 ## N_OSO STAB
      n_sect:          3
      n_desc:          1
      n_value:         1651001352
    - n_strx:          1
      n_type:          0x2E ## N_BNSYM STAB
      n_sect:          1
      n_desc:          0
      n_value:         0
    - n_strx:          41
      n_type:          0x24 ## N_FUN STAB
      n_sect:          1
      n_desc:          0
      n_value:         0
    - n_strx:          1
      n_type:          0x24 ## N_FUN STAB
      n_sect:          0
      n_desc:          0
      n_value:         18
    - n_strx:          1
      n_type:          0x4E ## N_ENSYM STAB
      n_sect:          1
      n_desc:          0
      n_value:         18
    - n_strx:          1
      n_type:          0x64 ## N_SO STAB
      n_sect:          1
      n_desc:          0
      n_value:         0
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         0
  StringTable:
    - ' '
    - _main
    - '/tmp/'
    - test.c
    - '/private/tmp/test.o'
    - _main
    - ''
...
