// RUN: %clang %cflags -Wl,--entry=_custom_start -march=armv8.3-a %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck -check-prefix=PACRET %s
// RUN: llvm-bolt-binary-analysis --scanners=pauth  %t.exe 2>&1 | FileCheck %s

// PACRET-NOT: untrusted link register found before tail call

        .text

        .globl  callee
        .type   callee,@function
callee:
        ret
        .size callee, .-callee

        .globl  good_direct_tailcall_no_clobber
        .type   good_direct_tailcall_no_clobber,@function
good_direct_tailcall_no_clobber:
// CHECK-NOT: good_direct_tailcall_no_clobber
        b       callee
        .size good_direct_tailcall_no_clobber, .-good_direct_tailcall_no_clobber

        .globl  good_plt_tailcall_no_clobber
        .type   good_plt_tailcall_no_clobber,@function
good_plt_tailcall_no_clobber:
// CHECK-NOT: good_plt_tailcall_no_clobber
        b       callee_ext
        .size good_plt_tailcall_no_clobber, .-good_plt_tailcall_no_clobber

        .globl  good_indirect_tailcall_no_clobber
        .type   good_indirect_tailcall_no_clobber,@function
good_indirect_tailcall_no_clobber:
// CHECK-NOT: good_indirect_tailcall_no_clobber
        autia   x0, x1
        br      x0
        .size good_indirect_tailcall_no_clobber, .-good_indirect_tailcall_no_clobber

        .globl  bad_direct_tailcall_not_auted
        .type   bad_direct_tailcall_not_auted,@function
bad_direct_tailcall_not_auted:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_direct_tailcall_not_auted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   b       callee # TAILCALL
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        b       callee
        .size bad_direct_tailcall_not_auted, .-bad_direct_tailcall_not_auted

        .globl  bad_plt_tailcall_not_auted
        .type   bad_plt_tailcall_not_auted,@function
bad_plt_tailcall_not_auted:
// FIXME: Calls via PLT are disassembled incorrectly. Nevertheless, they are
//        still detected as tail calls.
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_plt_tailcall_not_auted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       bad_indirect_tailcall_not_auted # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   b       bad_indirect_tailcall_not_auted # TAILCALL
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        b       callee_ext
        .size bad_plt_tailcall_not_auted, .-bad_plt_tailcall_not_auted

        .globl  bad_indirect_tailcall_not_auted
        .type   bad_indirect_tailcall_not_auted,@function
bad_indirect_tailcall_not_auted:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_indirect_tailcall_not_auted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autia   x0, x1
        br      x0
        .size bad_indirect_tailcall_not_auted, .-bad_indirect_tailcall_not_auted

        .globl  bad_direct_tailcall_untrusted
        .type   bad_direct_tailcall_untrusted,@function
bad_direct_tailcall_untrusted:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_direct_tailcall_untrusted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_direct_tailcall_untrusted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   b       callee # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        b       callee
        .size bad_direct_tailcall_untrusted, .-bad_direct_tailcall_untrusted

        .globl  bad_plt_tailcall_untrusted
        .type   bad_plt_tailcall_untrusted,@function
bad_plt_tailcall_untrusted:
// FIXME: Calls via PLT are disassembled incorrectly. Nevertheless, they are
//        still detected as tail calls.
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_plt_tailcall_untrusted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       bad_indirect_tailcall_untrusted # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_plt_tailcall_untrusted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      b       bad_indirect_tailcall_untrusted # TAILCALL
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   b       bad_indirect_tailcall_untrusted # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        b       callee_ext
        .size bad_plt_tailcall_untrusted, .-bad_plt_tailcall_untrusted

        .globl  bad_indirect_tailcall_untrusted
        .type   bad_indirect_tailcall_untrusted,@function
bad_indirect_tailcall_untrusted:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_indirect_tailcall_untrusted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_indirect_tailcall_untrusted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        autia   x0, x1
        br      x0
        .size bad_indirect_tailcall_untrusted, .-bad_indirect_tailcall_untrusted

        .globl  good_direct_tailcall_trusted
        .type   good_direct_tailcall_trusted,@function
good_direct_tailcall_trusted:
// CHECK-NOT: good_direct_tailcall_trusted
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        b       callee
        .size good_direct_tailcall_trusted, .-good_direct_tailcall_trusted

        .globl  good_plt_tailcall_trusted
        .type   good_plt_tailcall_trusted,@function
good_plt_tailcall_trusted:
// CHECK-NOT: good_plt_tailcall_trusted
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        b       callee_ext
        .size good_plt_tailcall_trusted, .-good_plt_tailcall_trusted

        .globl  good_indirect_tailcall_trusted
        .type   good_indirect_tailcall_trusted,@function
good_indirect_tailcall_trusted:
// CHECK-NOT: good_indirect_tailcall_trusted
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        autia   x0, x1
        br      x0
        .size good_indirect_tailcall_trusted, .-good_indirect_tailcall_trusted

        .globl  good_direct_tailcall_no_clobber_multi_bb
        .type   good_direct_tailcall_no_clobber_multi_bb,@function
good_direct_tailcall_no_clobber_multi_bb:
// CHECK-NOT: good_direct_tailcall_no_clobber_multi_bb
        b 1f
1:
        b       callee
        .size good_direct_tailcall_no_clobber_multi_bb, .-good_direct_tailcall_no_clobber_multi_bb

        .globl  good_indirect_tailcall_no_clobber_multi_bb
        .type   good_indirect_tailcall_no_clobber_multi_bb,@function
good_indirect_tailcall_no_clobber_multi_bb:
// CHECK-NOT: good_indirect_tailcall_no_clobber_multi_bb
        autia   x0, x1
        b 1f
1:
        br      x0
        .size good_indirect_tailcall_no_clobber_multi_bb_multi_bb, .-good_indirect_tailcall_no_clobber_multi_bb_multi_bb

        .globl  bad_direct_tailcall_not_auted_multi_bb
        .type   bad_direct_tailcall_not_auted_multi_bb,@function
bad_direct_tailcall_not_auted_multi_bb:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_direct_tailcall_not_auted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        cbz     x3, 1f
        autiasp
        ldr     w2, [x30]
1:
        b       callee
        .size bad_direct_tailcall_not_auted_multi_bb, .-bad_direct_tailcall_not_auted_multi_bb

        .globl  bad_indirect_tailcall_not_auted_multi_bb
        .type   bad_indirect_tailcall_not_auted_multi_bb,@function
bad_indirect_tailcall_not_auted_multi_bb:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_indirect_tailcall_not_auted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        cbz     x3, 1f
        autiasp
        ldr     w2, [x30]
1:
        autia   x0, x1
        br      x0
        .size bad_indirect_tailcall_not_auted_multi_bb, .-bad_indirect_tailcall_not_auted_multi_bb

        .globl  bad_direct_tailcall_untrusted_multi_bb
        .type   bad_direct_tailcall_untrusted_multi_bb,@function
bad_direct_tailcall_untrusted_multi_bb:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_direct_tailcall_untrusted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_direct_tailcall_untrusted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      b       callee # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        cbz     x3, 1f
        ldr     w2, [x30]
1:
        b       callee
        .size bad_direct_tailcall_untrusted_multi_bb, .-bad_direct_tailcall_untrusted_multi_bb

        .globl  bad_indirect_tailcall_untrusted_multi_bb
        .type   bad_indirect_tailcall_untrusted_multi_bb,@function
bad_indirect_tailcall_untrusted_multi_bb:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_indirect_tailcall_untrusted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_indirect_tailcall_untrusted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        cbz     x3, 1f
        ldr     w2, [x30]
1:
        autia   x0, x1
        br      x0
        .size bad_indirect_tailcall_untrusted_multi_bb, .-bad_indirect_tailcall_untrusted_multi_bb

        .globl  good_direct_tailcall_trusted_multi_bb
        .type   good_direct_tailcall_trusted_multi_bb,@function
good_direct_tailcall_trusted_multi_bb:
// CHECK-NOT: good_direct_tailcall_trusted_multi_bb
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        b 1f
1:
        b       callee
        .size good_direct_tailcall_trusted_multi_bb, .-good_direct_tailcall_trusted_multi_bb

        .globl  good_indirect_tailcall_trusted_multi_bb
        .type   good_indirect_tailcall_trusted_multi_bb,@function
good_indirect_tailcall_trusted_multi_bb:
// CHECK-NOT: good_indirect_tailcall_trusted_multi_bb
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        b 1f
1:
        autia   x0, x1
        br      x0
        .size good_indirect_tailcall_trusted_multi_bb, .-good_indirect_tailcall_trusted_multi_bb

        .globl  good_direct_tailcall_no_clobber_nocfg
        .type   good_direct_tailcall_no_clobber_nocfg,@function
good_direct_tailcall_no_clobber_nocfg:
// CHECK-NOT: good_direct_tailcall_no_clobber_nocfg
        adr     x3, 1f
        br      x3
1:
        b       callee
        .size good_direct_tailcall_no_clobber_nocfg, .-good_direct_tailcall_no_clobber_nocfg

        .globl  good_plt_tailcall_no_clobber_nocfg
        .type   good_plt_tailcall_no_clobber_nocfg,@function
good_plt_tailcall_no_clobber_nocfg:
// CHECK-NOT: good_plt_tailcall_no_clobber_nocfg
        adr     x3, 1f
        br      x3
1:
        b       callee_ext
        .size good_plt_tailcall_no_clobber_nocfg, .-good_plt_tailcall_no_clobber_nocfg

        .globl  good_indirect_tailcall_no_clobber_nocfg
        .type   good_indirect_tailcall_no_clobber_nocfg,@function
good_indirect_tailcall_no_clobber_nocfg:
// CHECK-NOT: good_indirect_tailcall_no_clobber_nocfg
        adr     x3, 1f
        br      x3
1:
        autia   x0, x1
        br      x0
        .size good_indirect_tailcall_no_clobber_nocfg, .-good_indirect_tailcall_no_clobber_nocfg

        .globl  bad_direct_tailcall_not_auted_nocfg
        .type   bad_direct_tailcall_not_auted_nocfg,@function
bad_direct_tailcall_not_auted_nocfg:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_direct_tailcall_not_auted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        b       callee
        .size bad_direct_tailcall_not_auted_nocfg, .-bad_direct_tailcall_not_auted_nocfg

        .globl  bad_plt_tailcall_not_auted_nocfg
        .type   bad_plt_tailcall_not_auted_nocfg,@function
bad_plt_tailcall_not_auted_nocfg:
// FIXME: Calls via PLT are disassembled incorrectly. Nevertheless, they are
//        still detected as tail calls.
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_plt_tailcall_not_auted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       bad_indirect_tailcall_not_auted_nocfg # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        b       callee_ext
        .size bad_plt_tailcall_not_auted_nocfg, .-bad_plt_tailcall_not_auted_nocfg

        .globl  bad_indirect_tailcall_not_auted_nocfg
        .type   bad_indirect_tailcall_not_auted_nocfg,@function
bad_indirect_tailcall_not_auted_nocfg:
// Known false positive: ignoring UNKNOWN CONTROL FLOW without CFG.
// CHECK-NOT: bad_indirect_tailcall_not_auted_nocfg
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autia   x0, x1
        br      x0
        .size bad_indirect_tailcall_not_auted_nocfg, .-bad_indirect_tailcall_not_auted_nocfg

        .globl  bad_direct_tailcall_untrusted_nocfg
        .type   bad_direct_tailcall_untrusted_nocfg,@function
bad_direct_tailcall_untrusted_nocfg:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_direct_tailcall_untrusted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_direct_tailcall_untrusted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      b       callee # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autiasp
        b       callee
        .size bad_direct_tailcall_untrusted_nocfg, .-bad_direct_tailcall_untrusted_nocfg

        .globl  bad_plt_tailcall_untrusted_nocfg
        .type   bad_plt_tailcall_untrusted_nocfg,@function
bad_plt_tailcall_untrusted_nocfg:
// FIXME: Calls via PLT are disassembled incorrectly. Nevertheless, they are
//        still detected as tail calls.
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_plt_tailcall_untrusted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       bad_indirect_tailcall_untrusted_nocfg # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_plt_tailcall_untrusted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      b       bad_indirect_tailcall_untrusted_nocfg # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autiasp
        b       callee_ext
        .size bad_plt_tailcall_untrusted_nocfg, .-bad_plt_tailcall_untrusted_nocfg

        .globl  bad_indirect_tailcall_untrusted_nocfg
        .type   bad_indirect_tailcall_untrusted_nocfg,@function
bad_indirect_tailcall_untrusted_nocfg:
// Known false negative: ignoring UNKNOWN CONTROL FLOW without CFG.
// Authentication oracle is found by a generic checker, though.
// CHECK-NOT: untrusted link register{{.*}}bad_indirect_tailcall_untrusted_nocfg
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_indirect_tailcall_untrusted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
// CHECK-NOT: untrusted link register{{.*}}bad_indirect_tailcall_untrusted_nocfg
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autiasp
        autia   x0, x1
        br      x0
        .size bad_indirect_tailcall_untrusted_nocfg, .-bad_indirect_tailcall_untrusted_nocfg

        .globl  good_direct_tailcall_trusted_nocfg
        .type   good_direct_tailcall_trusted_nocfg,@function
good_direct_tailcall_trusted_nocfg:
// CHECK-NOT: good_direct_tailcall_trusted_nocfg
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        b       callee
        .size good_direct_tailcall_trusted_nocfg, .-good_direct_tailcall_trusted_nocfg

        .globl  good_plt_tailcall_trusted_nocfg
        .type   good_plt_tailcall_trusted_nocfg,@function
good_plt_tailcall_trusted_nocfg:
// CHECK-NOT: good_plt_tailcall_trusted_nocfg
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        b       callee_ext
        .size good_plt_tailcall_trusted_nocfg, .-good_plt_tailcall_trusted_nocfg

        .globl  good_indirect_tailcall_trusted_nocfg
        .type   good_indirect_tailcall_trusted_nocfg,@function
good_indirect_tailcall_trusted_nocfg:
// CHECK-NOT: good_indirect_tailcall_trusted_nocfg
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        adr     x3, 1f
        br      x3
1:
        ldp     x29, x30, [sp], #0x10
        autiasp
        ldr     w2, [x30]
        autia   x0, x1
        br      x0
        .size good_indirect_tailcall_trusted_nocfg, .-good_indirect_tailcall_trusted_nocfg

// Check Armv8.3-a fused auth+branch instructions.

        .globl  good_indirect_tailcall_no_clobber_v83
        .type   good_indirect_tailcall_no_clobber_v83,@function
good_indirect_tailcall_no_clobber_v83:
// CHECK-NOT: good_indirect_tailcall_no_clobber_v83
        braa    x0, x1
        .size good_indirect_tailcall_no_clobber_v83, .-good_indirect_tailcall_no_clobber_v83

        .globl  bad_indirect_tailcall_untrusted_v83
        .type   bad_indirect_tailcall_untrusted_v83,@function
bad_indirect_tailcall_untrusted_v83:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_indirect_tailcall_untrusted_v83, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      braa    x0, x1 # TAILCALL
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_indirect_tailcall_untrusted_v83, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      braa    x0, x1 # TAILCALL
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   braa    x0, x1 # TAILCALL
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        autiasp
        braa    x0, x1
        .size bad_indirect_tailcall_untrusted_v83, .-bad_indirect_tailcall_untrusted_v83

// Make sure ELF entry function does not generate false positive reports.
// Additionally, check that the correct entry point is read from ELF header.

        .globl  _start
        .type   _start,@function
_start:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function _start, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      b       callee # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     x30, #0x0
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x30, #0x0
// CHECK-NEXT:  {{[0-9a-f]+}}:   b       callee # TAILCALL
        mov     x30, #0
        b       callee
        .size   _start, .-_start

        .globl  _custom_start
        .type   _custom_start,@function
_custom_start:
// CHECK-NOT: _custom_start
        mov     x30, #0
        b       callee
        .size   _custom_start, .-_custom_start

// Test two issues being reported for the same instruction.

        .globl  bad_non_protected_indirect_tailcall_not_auted
        .type   bad_non_protected_indirect_tailcall_not_auted,@function
bad_non_protected_indirect_tailcall_not_auted:
// CHECK-LABEL: GS-PAUTH: untrusted link register found before tail call in function bad_non_protected_indirect_tailcall_not_auted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x0, [x1]
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_non_protected_indirect_tailcall_not_auted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x1]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x0, [x1]
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        ldr     x0, [x1]
        br      x0
        .size bad_non_protected_indirect_tailcall_not_auted, .-bad_non_protected_indirect_tailcall_not_auted

        .globl  main
        .type   main,@function
main:
        mov x0, 0
        ret
        .size   main, .-main
