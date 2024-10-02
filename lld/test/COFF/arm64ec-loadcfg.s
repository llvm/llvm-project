# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.obj
# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig.obj

# RUN: lld-link -machine:arm64ec -dll -noentry %t.obj %t-loadconfig.obj -out:%t.dll

# RUN: llvm-readobj --coff-load-config %t.dll | FileCheck --check-prefix=LOADCFG %s
# LOADCFG:      CHPEMetadata [
# LOADCFG-NEXT:   Version: 0x2
# LOADCFG-NEXT:   CodeMap: 4096
# LOADCFG-NEXT:   CodeRangesToEntryPoints: 4096
# LOADCFG-NEXT:   RedirectionMetadata: 12288
# LOADCFG-NEXT:   __os_arm64x_dispatch_call_no_redirect: 0x1158
# LOADCFG-NEXT:   __os_arm64x_dispatch_ret: 0x1160
# LOADCFG-NEXT:   __os_arm64x_dispatch_call: 0x1168
# LOADCFG-NEXT:   __os_arm64x_dispatch_icall: 0x1170
# LOADCFG-NEXT:   __os_arm64x_dispatch_icall_cfg: 0x1188
# LOADCFG-NEXT:   AlternateEntryPoint: 0x0
# LOADCFG-NEXT:   AuxiliaryIAT: 0x0
# LOADCFG-NEXT:   GetX64InformationFunctionPointer: 0x1178
# LOADCFG-NEXT:   SetX64InformationFunctionPointer: 0x1180
# LOADCFG-NEXT:   ExtraRFETable: 0x0
# LOADCFG-NEXT:   ExtraRFETableSize: 0x0
# LOADCFG-NEXT:   __os_arm64x_dispatch_fptr: 0x1190
# LOADCFG-NEXT:   AuxiliaryIATCopy: 0x0
# LOADCFG-NEXT:   AuxiliaryDelayloadIAT: 0x0
# LOADCFG-NEXT:   AuxiliaryDelayloadIATCopy: 0x0
# LOADCFG-NEXT:   HybridImageInfoBitfield: 0x0
# LOADCFG-NEXT: ]

# RUN: llvm-readobj --hex-dump=.test %t.dll | FileCheck --check-prefix=TEST %s
# TEST: 0x180003000 00000000 00000000 00000000

.section .test,"dr"
        .rva __arm64x_native_entrypoint
        .rva __guard_check_icall_a64n_fptr
        .word __hybrid_image_info_bitfield
