# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --no-entry %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Check that "__llvm_covfun" custom section is aligned to 8 bytes.

        .section        .custom_section.__llvm_covfun,"GR",@,__covrec_A
        .int32  1
        .int8   2
# pad   .int8   0
#       .int8   0
#       .int8   0

        .section        .custom_section.__llvm_covfun,"GR",@,__covrec_B
        .int32  3

# CHECK:      - Type:            CUSTOM
# CHECK-NEXT:   Name:            __llvm_covfun
# CHECK-NEXT:   Payload:         '010000000200000003000000'

# Check that regular custom sections are not aligned.
        .section        .custom_section.foo,"GR",@,foo_A
        .int32  1
        .int8   2

        .section        .custom_section.foo,"GR",@,foo_B
        .int32  3

# CHECK:      - Type:            CUSTOM
# CHECK-NEXT:   Name:            foo
# CHECK-NEXT:   Payload:         '010000000203000000'
