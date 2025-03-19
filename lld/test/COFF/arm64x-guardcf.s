// REQUIRES: aarch64, x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows func-gfids.s -o func-gfids-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows func-gfids.s -o func-gfids-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows func-exp.s -o func-exp-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows func-exp.s -o func-exp-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows dllmain.s -o dllmain-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows dllmain.s -o dllmain-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows func-amd64.s -o func-amd64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj


// Check that CF guard tables contain both native and EC symbols and are referenced from both load configs.

// RUN: lld-link -dll -noentry -machine:arm64x func-gfids-arm64.obj func-gfids-arm64ec.obj func-amd64.obj -guard:cf -out:out.dll \
// RUN:          loadconfig-arm64ec.obj loadconfig-arm64.obj
// RUN: llvm-readobj --coff-load-config out.dll | FileCheck --check-prefix=LOADCFG %s

// LOADCFG:      LoadConfig [
// LOADCFG:        GuardCFFunctionCount: 3
// LOADCFG-NEXT:   GuardFlags [ (0x10500)
// LOADCFG-NEXT:     CF_FUNCTION_TABLE_PRESENT (0x400)
// LOADCFG-NEXT:     CF_INSTRUMENTED (0x100)
// LOADCFG-NEXT:     CF_LONGJUMP_TABLE_PRESENT (0x10000)
// LOADCFG-NEXT:   ]
// LOADCFG:      ]
// LOADCFG:      GuardFidTable [
// LOADCFG-NEXT:   0x180001000
// LOADCFG-NEXT:   0x180002000
// LOADCFG-NEXT:   0x180003000
// LOADCFG-NEXT: ]
// LOADCFG:      HybridObject {
// LOADCFG:        LoadConfig [
// LOADCFG:          GuardCFFunctionCount: 3
// LOADCFG-NEXT:     GuardFlags [ (0x10500)
// LOADCFG-NEXT:       CF_FUNCTION_TABLE_PRESENT (0x400)
// LOADCFG-NEXT:       CF_INSTRUMENTED (0x100)
// LOADCFG-NEXT:       CF_LONGJUMP_TABLE_PRESENT (0x10000)
// LOADCFG-NEXT:     ]
// LOADCFG:        ]
// LOADCFG:        GuardFidTable [
// LOADCFG-NEXT:     0x180001000
// LOADCFG-NEXT:     0x180002000
// LOADCFG-NEXT:     0x180003000
// LOADCFG-NEXT:   ]
// LOADCFG:      ]


// Check that exports from both views are present in CF guard tables.

// RUN: lld-link -dll -noentry -machine:arm64x func-exp-arm64.obj func-exp-arm64ec.obj -guard:cf -out:out-exp.dll \
// RUN:          loadconfig-arm64ec.obj loadconfig-arm64.obj
// RUN: llvm-readobj --coff-load-config out-exp.dll | FileCheck --check-prefix=LOADCFG %s


// Check that entry points from both views are present in CF guard tables.

// RUN: lld-link -dll -machine:arm64x dllmain-arm64.obj dllmain-arm64ec.obj -guard:cf -out:out-entry.dll \
// RUN:          loadconfig-arm64ec.obj loadconfig-arm64.obj
// RUN: llvm-readobj --coff-load-config out-entry.dll | FileCheck --check-prefix=LOADCFG %s


// Check that both load configs are marked as instrumented if any input object was built with /guard:cf.

// RUN: lld-link -dll -noentry -machine:arm64x func-gfids-arm64ec.obj -out:out-nocfg.dll \
// RUN:          loadconfig-arm64ec.obj loadconfig-arm64.obj

// RUN: llvm-readobj --coff-load-config out-nocfg.dll | FileCheck --check-prefix=LOADCFG-INST %s

// LOADCFG-INST:      LoadConfig [
// LOADCFG-INST:        GuardFlags [ (0x100)
// LOADCFG-INST-NEXT:     CF_INSTRUMENTED (0x100)
// LOADCFG-INST-NEXT:   ]
// LOADCFG-INST:      ]
// LOADCFG-INST:      HybridObject {
// LOADCFG-INST:        LoadConfig [
// LOADCFG-INST:          GuardFlags [ (0x100)
// LOADCFG-INST-NEXT:       CF_INSTRUMENTED (0x100)
// LOADCFG-INST-NEXT:     ]
// LOADCFG-INST:        ]
// LOADCFG-INST:      ]

#--- func-gfids.s
        .def @feat.00; .scl 3; .type 0; .endef
        .globl @feat.00
@feat.00 = 0x800

        .globl func
func:
        ret

        .section .gfids$y,"dr"
        .symidx func

#--- func-amd64.s
        .def @feat.00; .scl 3; .type 0; .endef
        .globl @feat.00
@feat.00 = 0x800

        .globl func_amd64
func_amd64:
        ret

        .section .gfids$y,"dr"
        .symidx func_amd64

#--- func-exp.s
        .def func; .scl 2; .type 32; .endef
        .globl func
func:
        ret

        .section .drectve
        .ascii "-export:func"

#--- dllmain.s
        .def _DllMainCRTStartup; .scl 2; .type 32; .endef
        .globl _DllMainCRTStartup
_DllMainCRTStartup:
        ret
