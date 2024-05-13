// Test that the runtime correctly interposes ___cxa_atexit.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t

  .section  __TEXT,__text,regular,pure_instructions
  .build_version macos, 12, 0 sdk_version 12, 0

# main registers the atexit and sets the test result to one.
  .globl  _main                           ; -- Begin function main
  .p2align  2
_main:                                  ; @main
; %bb.0:
  sub sp, sp, #32                     ; =32
  stp x29, x30, [sp, #16]             ; 16-byte Folded Spill
  add x29, sp, #16                    ; =16
  stur  wzr, [x29, #-4]
  adrp  x8, __ZGVZ4mainE6onExit@PAGE
  add x8, x8, __ZGVZ4mainE6onExit@PAGEOFF
  ldarb w8, [x8]
  and w8, w8, #0x1
  and w8, w8, #0xff
  cbnz  w8, LBB0_3
; %bb.1:
  adrp  x0, __ZGVZ4mainE6onExit@PAGE
  add x0, x0, __ZGVZ4mainE6onExit@PAGEOFF
  bl  ___cxa_guard_acquire
  cbz w0, LBB0_3
; %bb.2:
  adrp  x0, __ZN6OnExitD1Ev@GOTPAGE
  ldr x0, [x0, __ZN6OnExitD1Ev@GOTPAGEOFF]
  adrp  x1, __ZZ4mainE6onExit@PAGE
  add x1, x1, __ZZ4mainE6onExit@PAGEOFF
  adrp  x2, ___dso_handle@PAGE
  add x2, x2, ___dso_handle@PAGEOFF
  bl  ___cxa_atexit
  adrp  x0, __ZGVZ4mainE6onExit@PAGE
  add x0, x0, __ZGVZ4mainE6onExit@PAGEOFF
  bl  ___cxa_guard_release
LBB0_3:
  mov x0, #1
  bl  _llvm_jitlink_setTestResultOverride
  mov w0, #0
  ldp x29, x30, [sp, #16]             ; 16-byte Folded Reload
  add sp, sp, #32                     ; =32
  ret
                                        ; -- End function

# OnExit destructor resets the test result override to zero.
  .globl  __ZN6OnExitD1Ev                 ; -- Begin function _ZN6OnExitD1Ev
  .weak_def_can_be_hidden __ZN6OnExitD1Ev
  .p2align  2
__ZN6OnExitD1Ev:                        ; @_ZN6OnExitD1Ev
; %bb.0:
  sub sp, sp, #32                     ; =32
  stp x29, x30, [sp, #16]             ; 16-byte Folded Spill
  add x29, sp, #16                    ; =16
  str x0, [sp, #8]
  ldr x0, [sp, #8]
  str x0, [sp]                        ; 8-byte Folded Spill
  bl  __ZN6OnExitD2Ev
  ldr x0, [sp]                        ; 8-byte Folded Reload
  ldp x29, x30, [sp, #16]             ; 16-byte Folded Reload
  add sp, sp, #32                     ; =32
  ret
                                        ; -- End function
  .globl  __ZN6OnExitD2Ev                 ; -- Begin function _ZN6OnExitD2Ev
  .weak_def_can_be_hidden __ZN6OnExitD2Ev
  .p2align  2
__ZN6OnExitD2Ev:                        ; @_ZN6OnExitD2Ev
; %bb.0:
  sub sp, sp, #32                     ; =32
  stp x29, x30, [sp, #16]             ; 16-byte Folded Spill
  add x29, sp, #16                    ; =16
  str x0, [sp, #8]
  ldr x8, [sp, #8]
  str x8, [sp]                        ; 8-byte Folded Spill
  mov x0, #0
  bl  _llvm_jitlink_setTestResultOverride
  ldr x0, [sp]                        ; 8-byte Folded Reload
  ldp x29, x30, [sp, #16]             ; 16-byte Folded Reload
  add sp, sp, #32                     ; =32
  ret
                                        ; -- End function
.zerofill __DATA,__bss,__ZZ4mainE6onExit,1,0 ; @_ZZ4mainE6onExit
.zerofill __DATA,__bss,__ZGVZ4mainE6onExit,8,3 ; @_ZGVZ4mainE6onExit
.subsections_via_symbols
