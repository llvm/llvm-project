// Verify that XFIXUP_GUEST (0x1000) is set in PEFIXUP records for
// ARM64X hybrid images when either source or target chunk lives in a
// different machine than the image's primary machine (ARM64).

// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows host.s -o host.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows guest.s -o guest.obj
// RUN: rm -f out.dll out.pdb
// RUN: lld-link -machine:arm64x -dll -noentry -debug -debugtype:cv,fixup \
// RUN:   -out:out.dll -pdb:out.pdb host.obj guest.obj
// RUN: llvm-pdbutil dump --fixups out.pdb | FileCheck %s

// host.obj contributes one ADDR32NB in .data targeting a host (ARM64)
// function -> wSpare = 0x4000 CODE (no GUEST; no CV debug means the
// target isn't promoted to FUNCTION).
// guest.obj (ARM64EC, guest arch) contributes one ADDR32NB in .data
// targeting its own EC function -> source chunk machine = ARM64EC,
// flagged as guest -> wSpare = 0x5000 CODE|GUEST.
// CHECK:      Fixup Data
// CHECK:      | Type | Flags  | Tag
// CHECK-DAG:  |    2 | 0x4000 | CODE
// CHECK-DAG:  |    2 | 0x5000 | CODE GUEST

#--- host.s
        .section .text,"xr",discard,host_fn
        .globl host_fn
        .p2align 2
host_fn:
        mov w0, #1
        ret

        .data
        .globl pHost
pHost:
        .rva host_fn

#--- guest.s
        .section .text,"xr",discard,"#guest_fn"
        .globl "#guest_fn"
        .p2align 2
"#guest_fn":
        mov w0, #2
        ret

        .data
        .globl pGuest
pGuest:
        .rva "#guest_fn"
