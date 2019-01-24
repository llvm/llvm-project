# RUN: yaml2obj < %s > %t.obj
# RUN: not lld-link /include:symbol /dll /noentry /nodefaultlib %t.obj /out:%t.exe 2>&1 | FileCheck %s

# Tests that an associative comdat being associated with another
# associated comdat later in the file produces an error.
# CHECK: lld-link: error: {{.*}}: associative comdat .text$ac1 has invalid reference to section .text$ac2
# CHECK-NOT: lld-link: error:

--- !COFF
header:          
  Machine:         IMAGE_FILE_MACHINE_AMD64
  Characteristics: [  ]
sections:        
  - Name:            '.text$ac1'
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE ]
    Alignment:       1
    SectionData:     '01000000'
  - Name:            '.text$ac2'
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE ]
    Alignment:       1
    SectionData:     '01000000'
  - Name:            '.text$nm'
    Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_LNK_COMDAT, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE ]
    Alignment:       1
    SectionData:     '01000000'
symbols:         
  - Name:            '.text$ac1'
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition: 
      Length:          4
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        3099354981
      Number:          2
      Selection:       IMAGE_COMDAT_SELECT_ASSOCIATIVE
  - Name:            '.text$ac2'
    Value:           0
    SectionNumber:   2
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition: 
      Length:          4
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        3099354981
      Number:          3
      Selection:       IMAGE_COMDAT_SELECT_ASSOCIATIVE
  - Name:            '.text$nm'
    Value:           0
    SectionNumber:   3
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
    SectionDefinition: 
      Length:          4
      NumberOfRelocations: 0
      NumberOfLinenumbers: 0
      CheckSum:        3099354981
      Number:          4
      Selection:       IMAGE_COMDAT_SELECT_ANY
  - Name:            symbol
    Value:           0
    SectionNumber:   3
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_EXTERNAL
  - Name:            assocsym2
    Value:           0
    SectionNumber:   1
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
  - Name:            assocsym
    Value:           0
    SectionNumber:   2
    SimpleType:      IMAGE_SYM_TYPE_NULL
    ComplexType:     IMAGE_SYM_DTYPE_NULL
    StorageClass:    IMAGE_SYM_CLASS_STATIC
...

