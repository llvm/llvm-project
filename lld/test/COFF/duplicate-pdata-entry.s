# REQUIRES: x86
#
# Test that duplicate COMDAT .pdata entries are removed in createSections.
#
# Each .pdata COMDAT section is exactly 12 bytes (one RUNTIME_FUNCTION) with
# 3 IMAGE_REL_AMD64_ADDR32NB relocations at offsets 0, 4, 8 for BeginAddress,
# EndAddress, and UnwindInfoAddress respectively.
#
# removeDuplicatePdataChunks() identifies duplicates by comparing the resolved
# BeginAddress and EndAddress (relocs 0 and 1 plus their in-section addends).
# When ICF folds helper_b into helper_a, both .pdata entries resolve to the
# same [begin, end) range and the duplicate is removed.
#
# RUN: yaml2obj %s -o %t.obj
#
# When ICF is enabled, helper_b folds into helper_a, making the two .pdata
# entries duplicates. Only one survives = 12 bytes = 0xC.
# RUN: lld-link %t.obj -dll -noentry -out:%t.dll 2>&1 | FileCheck --check-prefix=WARN %s
# RUN: llvm-readobj --sections %t.dll | FileCheck --check-prefix=ICF %s
# WARN: warning: duplicate .pdata entry in .pdata$parent_b
# ICF:      Name: .pdata
# ICF-NEXT: VirtualSize: 0xC
#
# Without ICF, both helpers remain distinct, so both .pdata entries are
# unique = 24 bytes = 0x18.
# RUN: lld-link %t.obj -dll -noentry -out:%t_noicf.dll -opt:noicf
# RUN: llvm-readobj --sections %t_noicf.dll | FileCheck --check-prefix=NOICF %s
# NOICF:      Name: .pdata
# NOICF-NEXT: VirtualSize: 0x18

--- !COFF
header:
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: []
sections:
  # .text$parent_a (17 bytes) - kept alive via export, not ICF-foldable
  - Name:            '.text$parent_a'
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       16
    SectionData:     4883EC28B80200000048FFC04883C428C3
  # .text$parent_b (20 bytes) - different from parent_a
  - Name:            '.text$parent_b'
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       16
    SectionData:     4883EC30B80300000048FFC848FFC04883C430C3
  # .text$helper (14 bytes) - helper_a, identical to helper_b
  - Name:            '.text$helper'
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       16
    SectionData:     4883EC20B8010000004883C420C3
  # .text$helper (14 bytes) - helper_b, identical to helper_a (ICF folds this)
  - Name:            '.text$helper'
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment:       16
    SectionData:     4883EC20B8010000004883C420C3
  # .xdata for helper_a
  - Name:            .xdata
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    Alignment:       4
    SectionData:     '0104010034010000'
  # .xdata for helper_b (different unwind info, e.g. different frame register)
  - Name:            .xdata
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ ]
    Alignment:       4
    SectionData:     '0104010054010000'
  # .pdata$parent_a - 12-byte RUNTIME_FUNCTION, 3 relocs, associative to parent_a
  # BeginAddress addend = 0, EndAddress addend = 0x0E (function size 14)
  - Name:            '.pdata$parent_a'
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_READ ]
    Alignment:       4
    SectionData:     '000000000E00000000000000'
    Relocations:
      - VirtualAddress:  0
        SymbolName:      helper_a
        Type:            IMAGE_REL_AMD64_ADDR32NB
      - VirtualAddress:  4
        SymbolName:      helper_a
        Type:            IMAGE_REL_AMD64_ADDR32NB
      - VirtualAddress:  8
        SymbolName:      xdata_sym
        Type:            IMAGE_REL_AMD64_ADDR32NB
  # .pdata$parent_b - 12-byte RUNTIME_FUNCTION, 3 relocs, associative to parent_b
  # Same begin/end as parent_a after ICF, but different UnwindInfoAddress.
  # removeDuplicatePdataChunks() only checks begin/end, so this is still a dup.
  - Name:            '.pdata$parent_b'
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_READ ]
    Alignment:       4
    SectionData:     '000000000E00000000000000'
    Relocations:
      - VirtualAddress:  0
        SymbolName:      helper_b
        Type:            IMAGE_REL_AMD64_ADDR32NB
      - VirtualAddress:  4
        SymbolName:      helper_b
        Type:            IMAGE_REL_AMD64_ADDR32NB
      - VirtualAddress:  8
        SymbolName:      xdata_b_sym
        Type:            IMAGE_REL_AMD64_ADDR32NB
  # .drectve - export parent_a and parent_b (keeps them and their assoc alive)
  - Name:            .drectve
    Characteristics: [ IMAGE_SCN_LNK_INFO, IMAGE_SCN_LNK_REMOVE ]
    SectionData:     202D6578706F72743A706172656E745F61202D6578706F72743A706172656E745F62
symbols:
  - Name:            '.text$parent_a'
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          17
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        1111111111
      Number:          1
      Selection:       IMAGE_COMDAT_SELECT_ANY
  - Name:            parent_a
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            '.text$parent_b'
    Value:           0
    SectionNumber:   2
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          20
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        2222222222
      Number:          2
      Selection:       IMAGE_COMDAT_SELECT_ANY
  - Name:            parent_b
    Value:           0
    SectionNumber:   2
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            '.text$helper'
    Value:           0
    SectionNumber:   3
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          14
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        3333333333
      Number:          3
      Selection:       IMAGE_COMDAT_SELECT_ANY
  - Name:            helper_a
    Value:           0
    SectionNumber:   3
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            '.text$helper'
    Value:           0
    SectionNumber:   4
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          14
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        3333333333
      Number:          4
      Selection:       IMAGE_COMDAT_SELECT_ANY
  - Name:            helper_b
    Value:           0
    SectionNumber:   4
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_FUNCTION
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            xdata_sym
    Value:           0
    SectionNumber:   5
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
  - Name:            xdata_b_sym
    Value:           0
    SectionNumber:   6
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
  - Name:            '.pdata$parent_a'
    Value:           0
    SectionNumber:   7
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          12
      NumberOfRelocations: 3
      NumberOfLinenumbers: 0
      CheckSum:        0
      Number:          1
      Selection:       IMAGE_COMDAT_SELECT_ASSOCIATIVE
  - Name:            '.pdata$parent_b'
    Value:           0
    SectionNumber:   8
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition:
      Length:          12
      NumberOfRelocations: 3
      NumberOfLinenumbers: 0
      CheckSum:        0
      Number:          2
      Selection:       IMAGE_COMDAT_SELECT_ASSOCIATIVE
...
