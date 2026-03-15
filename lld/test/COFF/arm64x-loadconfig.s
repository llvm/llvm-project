// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows test.s -o test.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows chpe.s -o chpe.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows loadconfig.s -o loadconfig.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows loadconfig-ec.s -o loadconfig-ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows loadconfig-short.s -o loadconfig-short.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows loadconfig-short.s -o loadconfig-short-arm64ec.obj
// RUN: llvm-lib -machine:arm64x -out:loadconfig.lib loadconfig.obj loadconfig-ec.obj

// RUN: lld-link -machine:arm64x -out:out-warn.dll -dll -noentry test.obj \
// RUN:           2>&1 | FileCheck --check-prefixes=WARN-LOADCFG,WARN-EC-LOADCFG %s
// WARN-LOADCFG:    lld-link: warning: native version of '_load_config_used' is missing for ARM64X target
// WARN-EC-LOADCFG: lld-link: warning: EC version of '_load_config_used' is missing

// RUN: lld-link -machine:arm64x -out:out-nonative.dll -dll -noentry loadconfig-ec.obj chpe.obj \
// RUN:           2>&1 | FileCheck --check-prefixes=WARN-LOADCFG --implicit-check-not EC %s

// RUN: lld-link -machine:arm64ec -out:out-ec.dll -dll -noentry chpe.obj \
// RUN:           2>&1 | FileCheck --check-prefixes=WARN-EC-LOADCFG --implicit-check-not native %s

// RUN: lld-link -machine:arm64x -out:out.dll -dll -noentry loadconfig.obj test.obj \
// RUN:           2>&1 | FileCheck --check-prefixes=WARN-EC-LOADCFG --implicit-check-not native %s

// RUN: llvm-readobj --coff-load-config out.dll | FileCheck --check-prefix=DYNRELOCS %s
// DYNRELOCS:      DynamicValueRelocTableOffset: 0xC
// DYNRELOCS-NEXT: DynamicValueRelocTableSection: 4
// DYNRELOCS:      DynamicRelocations [
// DYNRELOCS-NEXT:   Version: 0x1
// DYNRELOCS-NEXT:   Arm64X [
// DYNRELOCS-NEXT:     Entry [
// DYNRELOCS-NEXT:       RVA: 0x7C
// DYNRELOCS-NEXT:       Type: VALUE
// DYNRELOCS-NEXT:       Size: 0x2
// DYNRELOCS-NEXT:       Value: 0x8664
// DYNRELOCS-NEXT:     ]
// DYNRELOCS-NEXT:     Entry [
// DYNRELOCS-NEXT:       RVA: 0x150
// DYNRELOCS-NEXT:       Type: VALUE
// DYNRELOCS-NEXT:       Size: 0x4
// DYNRELOCS-NEXT:       Value: 0x0
// DYNRELOCS-NEXT:     ]
// DYNRELOCS-NEXT:     Entry [
// DYNRELOCS-NEXT:       RVA: 0x154
// DYNRELOCS-NEXT:       Type: VALUE
// DYNRELOCS-NEXT:       Size: 0x4
// DYNRELOCS-NEXT:       Value: 0x0
// DYNRELOCS-NEXT:     ]
// DYNRELOCS-NEXT:   ]
// DYNRELOCS-NEXT: ]

// RUN: llvm-readobj --headers out.dll | FileCheck --check-prefix=HEADERS %s
// HEADERS:      BaseRelocationTableRVA: 0x4000
// HEADERS-NEXT: BaseRelocationTableSize: 0xC
// HEADERS:      LoadConfigTableRVA: 0x1000
// HEADERS-NEXT: LoadConfigTableSize: 0x140
// HEADERS:      Name: .reloc (2E 72 65 6C 6F 63 00 00)
// HEADERS-NEXT: VirtualSize: 0x38

// RUN: lld-link -machine:arm64x -out:out-short.dll -dll -noentry loadconfig-short.obj 2>&1 | FileCheck --check-prefix=WARN-RELOC-SIZE %s
// RUN: lld-link -machine:arm64x -out:out-short.dll -dll -noentry loadconfig-short-arm64ec.obj 2>&1 | FileCheck --check-prefix=WARN-RELOC-SIZE %s
// WARN-RELOC-SIZE: lld-link: warning: '_load_config_used' structure too small to include dynamic relocations

// Check that the CHPE metadata pointer is correctly copied from the EC load config to the native load config.

// RUN: lld-link -machine:arm64x -out:out-hyb.dll -dll -noentry loadconfig.obj loadconfig-ec.obj chpe.obj test.obj

// RUN: llvm-readobj --coff-load-config out-hyb.dll | FileCheck --check-prefix=LOADCFG %s
// LOADCFG:      Format: COFF-ARM64X
// LOADCFG-NEXT: Arch: aarch64
// LOADCFG-NEXT: AddressSize: 64bit
// LOADCFG-NEXT: LoadConfig [
// LOADCFG-NEXT:   Size: 0x140
// LOADCFG:      CHPEMetadata [
// LOADCFG-NEXT:   Version: 0x2
// LOADCFG:        RedirectionMetadata: 12288
// LOADCFG:        AlternateEntryPoint: 0x0
// LOADCFG-NEXT:   AuxiliaryIAT: 0x0
// LOADCFG-NEXT:   GetX64InformationFunctionPointer: 0x0
// LOADCFG-NEXT:   SetX64InformationFunctionPointer: 0x0
// LOADCFG-NEXT:   ExtraRFETable: 0x0
// LOADCFG-NEXT:   ExtraRFETableSize: 0x0
// LOADCFG-NEXT:   __os_arm64x_dispatch_fptr: 0x0
// LOADCFG-NEXT:   AuxiliaryIATCopy: 0x0
// LOADCFG-NEXT:   AuxiliaryDelayloadIAT: 0x0
// LOADCFG-NEXT:   AuxiliaryDelayloadIATCopy: 0x0
// LOADCFG-NEXT:   HybridImageInfoBitfield: 0x0
// LOADCFG:      ]
// LOADCFG-NEXT: DynamicRelocations [
// LOADCFG-NEXT:   Version: 0x1
// LOADCFG-NEXT:   Arm64X [
// LOADCFG-NEXT:     Entry [
// LOADCFG-NEXT:       RVA: 0x7C
// LOADCFG-NEXT:       Type: VALUE
// LOADCFG-NEXT:       Size: 0x2
// LOADCFG-NEXT:       Value: 0x8664
// LOADCFG-NEXT:     ]
// LOADCFG-NEXT:     Entry [
// LOADCFG-NEXT:       RVA: 0x150
// LOADCFG-NEXT:       Type: VALUE
// LOADCFG-NEXT:       Size: 0x4
// LOADCFG-NEXT:       Value: 0x1140
// LOADCFG-NEXT:     ]
// LOADCFG-NEXT:     Entry [
// LOADCFG-NEXT:       RVA: 0x154
// LOADCFG-NEXT:       Type: VALUE
// LOADCFG-NEXT:       Size: 0x4
// LOADCFG-NEXT:       Value: 0x140
// LOADCFG-NEXT:     ]
// LOADCFG-NEXT:   ]
// LOADCFG-NEXT: ]
// LOADCFG-NEXT: HybridObject {
// LOADCFG-NEXT:   Format: COFF-ARM64EC
// LOADCFG-NEXT:   Arch: aarch64
// LOADCFG-NEXT:   AddressSize: 64bit
// LOADCFG-NEXT:   LoadConfig [
// LOADCFG-NEXT:     Size:   0x140
// LOADCFG:        CHPEMetadata [
// LOADCFG-NEXT:     Version:   0x2
// LOADCFG:        ]
// LOADCFG-NEXT:   DynamicRelocations [
// LOADCFG-NEXT:     Version: 0x1
// LOADCFG-NEXT:     Arm64X [
// LOADCFG-NEXT:       Entry [
// LOADCFG-NEXT:         RVA: 0x7C
// LOADCFG-NEXT:         Type: VALUE
// LOADCFG-NEXT:         Size: 0x2
// LOADCFG-NEXT:         Value: 0x8664
// LOADCFG-NEXT:       ]
// LOADCFG-NEXT:       Entry [
// LOADCFG-NEXT:         RVA: 0x150
// LOADCFG-NEXT:         Type: VALUE
// LOADCFG-NEXT:         Size: 0x4
// LOADCFG-NEXT:         Value: 0x1140
// LOADCFG-NEXT:       ]
// LOADCFG-NEXT:       Entry [
// LOADCFG-NEXT:         RVA: 0x154
// LOADCFG-NEXT:         Type: VALUE
// LOADCFG-NEXT:         Size: 0x4
// LOADCFG-NEXT:         Value: 0x140
// LOADCFG-NEXT:       ]
// LOADCFG-NEXT:     ]
// LOADCFG-NEXT:   ]
// LOADCFG-NEXT: }

// RUN: llvm-readobj --coff-basereloc out-hyb.dll | FileCheck --check-prefix=BASERELOC %s
// BASERELOC:      BaseReloc [
// BASERELOC-NEXT:   Entry {
// BASERELOC-NEXT:     Type: DIR64
// BASERELOC-NEXT:     Address: 0x10C8
// BASERELOC-NEXT:   }
// BASERELOC-NEXT:   Entry {
// BASERELOC-NEXT:     Type: DIR64
// BASERELOC-NEXT:     Address: 0x1208
// BASERELOC-NEXT:   }
// BASERELOC-NEXT:   Entry {
// BASERELOC-NEXT:     Type: DIR64
// BASERELOC-NEXT:     Address: 0x2074
// BASERELOC-NEXT:   }

// RUN: lld-link -machine:arm64x -out:out-hyb-lib.dll -dll -noentry loadconfig.lib chpe.obj test.obj
// RUN: llvm-readobj --coff-load-config out-hyb-lib.dll | FileCheck --check-prefix=LOADCFG %s
// RUN: llvm-readobj --coff-basereloc out-hyb-lib.dll | FileCheck --check-prefix=BASERELOC %s

#--- test.s
        .data
sym:
        // Emit a basereloc to make the loadconfig test more meaningful.
        .xword sym

#--- loadconfig.s
        .section .rdata,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0x140
        .fill 0x13c,1,0

#--- loadconfig-ec.s
        .section .rdata,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0x140
        .fill 0xc4,1,0
        .xword __chpe_metadata
        .fill 0x70,1,0

#--- loadconfig-short.s
        .section .rdata,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0xe4
        .fill 0xe0,1,0

#--- chpe.s
        .data
        .globl __chpe_metadata
        .p2align 3, 0
__chpe_metadata:
        .word 2
        .rva __hybrid_code_map
        .word __hybrid_code_map_count
        .rva __x64_code_ranges_to_entry_points
        .rva __arm64x_redirection_metadata
        .word 0 // __os_arm64x_dispatch_call_no_redirect
        .word 0 // __os_arm64x_dispatch_ret
        .word 0 // __os_arm64x_check_call
        .word 0 // __os_arm64x_check_icall
        .word 0 // __os_arm64x_check_icall_cfg
        .rva __arm64x_native_entrypoint
        .rva __hybrid_auxiliary_iat
        .word __x64_code_ranges_to_entry_points_count
        .word __arm64x_redirection_metadata_count
        .word 0 // __os_arm64x_get_x64_information
        .word 0 // __os_arm64x_set_x64_information
        .rva __arm64x_extra_rfe_table
        .word __arm64x_extra_rfe_table_size
        .word 0 // __os_arm64x_dispatch_fptr
        .rva __hybrid_auxiliary_iat_copy
        .rva __hybrid_auxiliary_delayload_iat
        .rva __hybrid_auxiliary_delayload_iat_copy
        .word __hybrid_image_info_bitfield
        .word 0 // __os_arm64x_helper3
        .word 0 // __os_arm64x_helper4
        .word 0 // __os_arm64x_helper5
        .word 0 // __os_arm64x_helper6
        .word 0 // __os_arm64x_helper7
        .word 0 // __os_arm64x_helper8
