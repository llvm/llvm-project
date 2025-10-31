// REQUIRES: x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=x86_64-windows refab.s -o refab.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows aa.s -o aa.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows b.s -o b.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows antidep.s -o antidep.obj
// RUN: llvm-lib -out:aa.lib aa.obj
// RUN: llvm-lib -out:b.lib b.obj

// Check that -alternatename with an undefined target does not prevent the symbol from being resolved to a library,
// once another alternate name is resolved and pulls in the source symbol.
// RUN: lld-link -out:out.dll -dll -noentry -machine:amd64 refab.obj aa.lib -alternatename:a=aa -alternatename:b=undef

// Check that -alternatename with an anti-dependency target does not prevent the symbol from being resolved to a library,
// after another alternate name is resolved and pulls in the source symbol.
// RUN: lld-link -out:out2.dll -dll -noentry -machine:amd64 antidep.obj refab.obj aa.lib -alternatename:a=aa -alternatename:b=u

#--- refab.s
        .data
        .rva a
        .rva b

#--- aa.s
        .globl aa
aa:
        .word 1

        .section .drectve, "yn"
        .ascii "/defaultlib:b.lib"

#--- b.s
        .globl b
b:
        .word 2

#--- antidep.s
        .weak_anti_dep u
        .set u,d

        .globl d
d:
        .word 3
