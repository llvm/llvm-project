// Test that basic MachO TLVs work by adding together TLVs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (__thread_bss) and non-zero-initialized
// (__thread_data) secitons.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

  .section  __TEXT,__text,regular,pure_instructions
  .build_version macos, 12, 0 sdk_version 12, 0
  .globl  _main                           ; -- Begin function main
  .p2align  2
_main:                                  ; @main
  .cfi_startproc
; %bb.0:
  sub sp, sp, #32                     ; =32
  stp x29, x30, [sp, #16]             ; 16-byte Folded Spill
  add x29, sp, #16                    ; =16
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  adrp  x0, _x@TLVPPAGE
  ldr x0, [x0, _x@TLVPPAGEOFF]
  ldr x8, [x0]
  blr x8
  mov x8, x0
  adrp  x0, _y@TLVPPAGE
  ldr x0, [x0, _y@TLVPPAGEOFF]
  ldr x9, [x0]
  blr x9
  mov x9, x0
  adrp  x0, _z@TLVPPAGE
  ldr x0, [x0, _z@TLVPPAGEOFF]
  ldr x10, [x0]
  blr x10
  stur  wzr, [x29, #-4]
  ldr w8, [x8]
  ldr w9, [x9]
  add w8, w8, w9
  ldr w9, [x0]
  add w0, w8, w9
  ldp x29, x30, [sp, #16]             ; 16-byte Folded Reload
  add sp, sp, #32                     ; =32
  ret
  .cfi_endproc
                                        ; -- End function
.tbss _x$tlv$init, 4, 2                 ; @x

  .section  __DATA,__thread_vars,thread_local_variables
  .globl  _x
_x:
  .quad __tlv_bootstrap
  .quad 0
  .quad _x$tlv$init

  .section  __DATA,__thread_data,thread_local_regular
  .p2align  2                               ; @y
_y$tlv$init:
  .long 4294967295                      ; 0xffffffff

  .section  __DATA,__thread_vars,thread_local_variables
  .globl  _y
_y:
  .quad __tlv_bootstrap
  .quad 0
  .quad _y$tlv$init

  .section  __DATA,__thread_data,thread_local_regular
  .p2align  2                               ; @z
_z$tlv$init:
  .long 1                               ; 0x1

  .section  __DATA,__thread_vars,thread_local_variables
  .globl  _z
_z:
  .quad __tlv_bootstrap
  .quad 0
  .quad _z$tlv$init

.subsections_via_symbols
