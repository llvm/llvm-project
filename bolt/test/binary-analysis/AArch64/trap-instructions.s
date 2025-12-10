// RUN: %clang %cflags -march=armv8.3-a %s -o %t.exe -Wl,--emit-relocs
// RUN: llvm-bolt-binary-analysis --scanners=pauth %t.exe 2>&1 | FileCheck %s

// Test what instructions can be used to terminate the program abnormally
// on security violation.
//
// All test cases have the same structure:
//
//      cbz     x0, 1f    // [a], ensures [c] is never reported as unreachable
//      autia   x2, x3
//      cbz     x1, 2f    // [b]
//      [instruction under test]
// 1:
//      ret               // [c]
// 2:
//      ldr     x0, [x2]
//      ret
//
// This is to handle three possible cases: the instruction under test may be
// considered by BOLT as
// * trapping (and thus no-return): after being authenticated, x2 is ether
//   checked by LDR (if [b] is taken) or the program is terminated
//   immediately without leaking x2 (if [b] falls through to the trapping
//   instruction under test). Nothing is reported.
// * non-trapping, but no-return (such as calling abort()): x2 is leaked if [b]
//   falls through. Authentication oracle is reported.
// * non-trapping and falling-through (i.e. a regular instruction):
//   x2 is leaked by [c]. Authentication oracle is reported.

        .text

        .globl  brk_key_ia
        .type   brk_key_ia,@function
brk_key_ia:
// CHECK-NOT: brk_key_ia
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0xc470
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_key_ia, .-brk_key_ia

        .globl  brk_key_ib
        .type   brk_key_ib,@function
brk_key_ib:
// CHECK-NOT: brk_key_ib
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0xc471
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_key_ib, .-brk_key_ib

        .globl  brk_key_da
        .type   brk_key_da,@function
brk_key_da:
// CHECK-NOT: brk_key_da
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0xc472
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_key_da, .-brk_key_da

        .globl  brk_key_db
        .type   brk_key_db,@function
brk_key_db:
// CHECK-NOT: brk_key_db
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0xc473
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_key_db, .-brk_key_db

// The immediate operand of BRK instruction may indicate whether the instruction
// is intended to be a non-recoverable trap: for example, for this code
//
//     int test_trap(void) {
//       __builtin_trap();
//       return 42;
//     }
//     int test_debugtrap(void) {
//       __builtin_debugtrap();
//       return 42;
//     }
//
// Clang produces the following assembly:
//
//     test_trap:
//             brk     #0x1
//     test_debugtrap:
//             brk     #0xf000
//             mov     w0, #42
//             ret
//
// In GCC, __builtin_trap() uses "brk 0x3e8" (i.e. decimal 1000) and
// __builtin_debugtrap() is not supported.
//
// At the time of writing these test cases, any BRK instruction is considered
// no-return by BOLT, thus it ends its basic block and prevents falling through
// to the next BB.
// FIXME: Make BOLT handle __builtin_debugtrap() properly from the CFG point
//        of view.

        .globl  brk_gcc_builtin_trap
        .type   brk_gcc_builtin_trap,@function
brk_gcc_builtin_trap:
// CHECK-NOT: brk_gcc_builtin_trap
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0x3e8     // __builtin_trap()
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_gcc_builtin_trap, .-brk_gcc_builtin_trap

        .globl  brk_clang_builtin_trap
        .type   brk_clang_builtin_trap,@function
brk_clang_builtin_trap:
// CHECK-NOT: brk_clang_builtin_trap
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0x1       // __builtin_trap()
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_clang_builtin_trap, .-brk_clang_builtin_trap

        .globl  brk_clang_builtin_debugtrap
        .type   brk_clang_builtin_debugtrap,@function
brk_clang_builtin_debugtrap:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function brk_clang_builtin_debugtrap, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x2, x3
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0xf000    // __builtin_debugtrap()
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_clang_builtin_debugtrap, .-brk_clang_builtin_debugtrap

// Conservatively assume BRK with an unknown immediate operand as not suitable
// for terminating the program on security violation.
        .globl  brk_unknown_imm
        .type   brk_unknown_imm,@function
brk_unknown_imm:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function brk_unknown_imm, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x2, x3
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        brk     0x3572
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   brk_unknown_imm, .-brk_unknown_imm

// Conservatively assume calling the abort() function may be an unsafe way to
// terminate the program, as there is some amount of instructions that would
// be executed when the program state is already tampered with.
        .globl  call_abort_fn
        .type   call_abort_fn,@function
call_abort_fn:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function call_abort_fn, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x2, x3
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        cbz     x0, 1f
        autia   x2, x3
        cbz     x1, 2f
        b       abort    // a no-return tail call to abort()
1:
        ret
2:
        ldr     x0, [x2]
        ret
        .size   call_abort_fn, .-call_abort_fn

        .globl  main
        .type   main,@function
main:
        mov     x0, 0
        ret
        .size   main, .-main
