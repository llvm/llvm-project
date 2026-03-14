# RUN: mkdir -p %t.dir
# RUN: yaml2obj %s -o %t.dir/foo.obj

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver foo.obj
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=DRIVER %s

# DRIVER-NOT: IMAGE_FILE_UP_SYSTEM_ONLY
# DRIVER-NOT: IMAGE_DLL_CHARACTERISTICS_WDM_DRIVER
# DRIVER: AddressOfEntryPoint: 0x1000

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver:uponly foo.obj
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=UPONLY %s

# UPONLY: IMAGE_FILE_UP_SYSTEM_ONLY
# UPONLY: AddressOfEntryPoint: 0x1000

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver:wdm foo.obj
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=WDM %s

# WDM: AddressOfEntryPoint: 0x1004
# WDM: IMAGE_DLL_CHARACTERISTICS_WDM_DRIVER

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver:wdm,uponly foo.obj
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=BOTH %s

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver:uponly,wdm foo.obj
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=BOTH %s

# BOTH: IMAGE_FILE_UP_SYSTEM_ONLY
# BOTH: AddressOfEntryPoint: 0x1004
# BOTH: IMAGE_DLL_CHARACTERISTICS_WDM_DRIVER

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver foo.obj
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=FIXED1 %s

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver foo.obj /fixed:no
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=FIXED1 %s

# FIXED1: IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE

# RUN: rm -f %t.dir/foo.sys
# RUN: cd %t.dir; lld-link /driver foo.obj /fixed
# RUN: llvm-readobj --file-headers %t.dir/foo.sys | FileCheck -check-prefix=FIXED2 %s

# FIXED2-NOT: IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE

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
  - Name:            mainCRTStartup
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            _NtProcessStartup
    Value:           4
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
...
