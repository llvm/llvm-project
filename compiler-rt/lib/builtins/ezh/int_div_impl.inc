//===-- int_div_impl.inc - EZH Integer division/modulo helper --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(INT_DIV)
#define FUNCTION_NAME __udivsi3
#define RETURN_QUOTIENT
#elif defined(INT_MOD)
#define FUNCTION_NAME __umodsi3
#define RETURN_REMAINDER
#else
#error "Either INT_DIV or INT_MOD must be defined before including this file"
#endif

.text
.p2align 2

.globl FUNCTION_NAME
.type FUNCTION_NAME, @function
.hidden FUNCTION_NAME

FUNCTION_NAME:
    ; 1. Save callee-saved registers to stack
    e_pushd r4
    e_pushd r5
    
    ; 2. Setup loop parameters
    e_load_imm r5, 31             ; Loop count register = 31 (repetition count, runs 32 times total)
    e_load_imm r2, 0              ; R2 = Remainder (initialized to 0)
    
    ; 3. Setup hardware zero-overhead tight loop safely
    e_ldr r4, pc, 1f              ; SAFE: Load Rend absolute address from constant island (bypasses broken PC GPR read!)
    e_tight_loop r4, r5           ; Enable tight loop supervisor (executes R5+1 times)
    e_nop                         ; Hardware Quirk: The instruction immediately following e_tight_loop
                                  ; only executes ONCE at initial entry. This NOP absorbs that cycle.
    
    ; --- Loop Body (32 iterations, exactly 6 cycles, no branching overhead) ---
    e_lsls r0, r0, 1              ; 1. Shift dividend R0 left by 1, the MSB shifts out into Carry flag
    e_lsl r2, r2, 1               ; 2. Shift remainder R2 left by 1 (making room for shifted-out bit)
    e_bset_imm_ca r2, r2, 0       ; 3. Shift Carry (the old MSB of R0) into the LSB of remainder R2
    e_subs r3, r2, r1             ; 4. Compare: R3 = R2 - R1, updates Carry flag with unsigned borrow out of MSB
                                  ;    (No Borrow -> R2 >= R1 -> Carry is 0 -> NC flag is set to 1)
    e_mov_nc r2, r3               ; 5. Conditional Restore: R2 = R3 if R2 >= R1 (NC flag is 1)
    e_bset_imm_nc r0, r0, 0       ; 6. Set Quotient Bit: R0[0] = 1 if R2 >= R1 (NC flag is 1)
    ; --- Loop Ends Here ---
    
loop_end_label:
#if defined(RETURN_REMAINDER)
    e_mov r0, r2                  ; Modulo: Return remainder in R0
#endif
    ; 4. Restore registers and return
    e_popd r5
    e_popd r4
    e_mov pc, ra

.p2align 2
1:  .long loop_end_label

.size FUNCTION_NAME, . - FUNCTION_NAME
