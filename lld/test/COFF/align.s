# RUN: yaml2obj < %s > %t.obj
# RUN: lld-link /out:%t.exe /entry:main /align:32 %t.obj
# RUN: llvm-readobj --file-headers %t.exe | FileCheck %s

# CHECK: SectionAlignment: 32

# RUN: lld-link /out:%t.exe /entry:main /align:32 %t.obj 2>&1 \
# RUN:   | FileCheck -check-prefix=WARN1 %s

# WARN1: /align specified without /driver; image may not run

# RUN: lld-link /out:%t.exe /entry:main /align:32 %t.obj /driver 2>&1 \
# RUN:   | FileCheck -check-prefix=WARN2 --allow-empty %s

# RUN: lld-link /out:%t.exe /entry:main %t.obj /driver 2>&1 \
# RUN:   | FileCheck -check-prefix=WARN2 --allow-empty %s

# WARN2-NOT: /align specified without /driver; image may not run

--- !COFF
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: []
sections:
  - Name:            .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       4096
    SectionData:     0000000000000000
    Relocations:
      - VirtualAddress:  0
        SymbolName:      __ImageBase
        Type:            IMAGE_REL_AMD64_ADDR64
symbols:
  - Name:            .text
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          8
      NumberOfRelocations: 1
      NumberOfLinenumbers: 0
      CheckSum:        0
      Number:          0
  - Name:            main
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            __ImageBase
    Value:           0
    SectionNumber:   0
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
...
