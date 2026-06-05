; RUN: llc -filetype=obj %p/Inputs/many-funcs.ll -o %t.many.o
; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -r -o %t.wasm %t.many.o %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

; Test that relocations within the CODE section correctly handle
; linking object with different header sizes.  many-funcs.ll has
; 128 function and so the final output requires a 2-byte LEB in
; the CODE section header to store the function count.

target triple = "wasm32-unknown-unknown"

define i32 @func() {
entry:
  %call = tail call i32 @func()
  ret i32 %call
}

; CHECK:        - Type:            CODE
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x14
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x20
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x38
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x44
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x50
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x68
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x74
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x80
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x8C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x98
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xA4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xB0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xBC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xC8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xD4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xE0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xEC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0xF8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x104
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x110
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x11C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x128
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x134
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x140
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x14C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x158
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x164
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x170
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x17C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x188
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x194
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1A0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1AC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1B8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1C4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1D0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1DC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1E8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x1F4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x200
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x20C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x218
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x224
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x230
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x23C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x248
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x254
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x260
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x26C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x278
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x284
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x290
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x29C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2A8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2B4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2C0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2CC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2D8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2E4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2F0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x2FC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x308
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x314
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x320
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x32C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x338
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x344
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x350
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x35C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x368
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x374
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x380
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x38C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x398
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3A4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3B0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3BC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3C8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3D4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3E0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3EC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x3F8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x404
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x410
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x41C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x428
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x434
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x440
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x44C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x458
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x464
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x470
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x47C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x488
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x494
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4A0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4AC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4B8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4C4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4D0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4DC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4E8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4F4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x500
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x50C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x518
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x524
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x530
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x53C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x548
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x554
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x560
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x56C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x578
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x584
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x590
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x59C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5A8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5B4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5C0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5CC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5D8
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5E4
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x5F0
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           129
; CHECK-NEXT:         Offset:          0x5FC
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
; CHECK-NEXT:         Index:           129
; CHECK-NEXT:         Offset:          0x608
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           131
; CHECK-NEXT:         Offset:          0x611
