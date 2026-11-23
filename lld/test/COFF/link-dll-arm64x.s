REQUIRES: aarch64, x86
RUN: split-file %s %t.dir && cd %t.dir

RUN: llvm-mc -filetype=obj -triple=arm64ec-windows sym-ec.s -o sym-ec.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows ref-ec.s -o ref-ec.obj
RUN: llvm-mc -filetype=obj -triple=aarch64-windows sym-native.s -o sym-native.obj
RUN: llvm-mc -filetype=obj -triple=aarch64-windows ref-native.s -o ref-native.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj
RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj

RUN: lld-link -machine:arm64x -dll -noentry -out:import.dll sym-ec.obj sym-native.obj loadconfig-arm64ec.obj loadconfig-arm64.obj

RUN: lld-link -machine:arm64 -dll -noentry -out:out-arm64.dll ref-native.obj import.dll \
RUN:          -lldmingw -exclude-all-symbols -auto-import:no

RUN: llvm-readobj --coff-imports out-arm64.dll | FileCheck --check-prefix=NATIVE %s
NATIVE:      Import {
NATIVE-NEXT:   Name: import.dll
NATIVE-NEXT:   ImportLookupTableRVA:
NATIVE-NEXT:   ImportAddressTableRVA:
NATIVE-NEXT:   Symbol: native_data (0)
NATIVE-NEXT: }

RUN: lld-link -machine:arm64ec -dll -noentry -out:out-arm64ec.dll ref-ec.obj import.dll \
RUN:          loadconfig-arm64ec.obj -lldmingw -exclude-all-symbols -auto-import:no

RUN: llvm-readobj --coff-imports out-arm64ec.dll | FileCheck --check-prefix=EC %s
EC:      Import {
EC-NEXT:   Name: import.dll
EC-NEXT:   ImportLookupTableRVA:
EC-NEXT:   ImportAddressTableRVA:
EC-NEXT:   Symbol: ec_data (0)
EC-NEXT: }

RUN: lld-link -machine:arm64x -dll -noentry -out:out-arm64x.dll ref-ec.obj ref-native.obj import.dll \
RUN:          loadconfig-arm64ec.obj loadconfig-arm64.obj -lldmingw -exclude-all-symbols -auto-import:no

RUN: llvm-readobj --coff-imports out-arm64x.dll | FileCheck --check-prefix=ARM64X %s
ARM64X:      Import {
ARM64X-NEXT:   Name: import.dll
ARM64X-NEXT:   ImportLookupTableRVA:
ARM64X-NEXT:   ImportAddressTableRVA:
ARM64X-NEXT:   Symbol: native_data (0)
ARM64X-NEXT: }
ARM64X-NEXT: HybridObject {
ARM64X-NEXT:   Format: COFF-ARM64EC
ARM64X-NEXT:   Arch: aarch64
ARM64X-NEXT:   AddressSize: 64bit
ARM64X-NEXT:   Import {
ARM64X-NEXT:     Name: import.dll
ARM64X-NEXT:     ImportLookupTableRVA:
ARM64X-NEXT:     ImportAddressTableRVA:
ARM64X-NEXT:     Symbol: ec_data (0)
ARM64X-NEXT:   }
ARM64X-NEXT: }

#--- sym-ec.s
    .data
    .globl ec_data
ec_data:
    .word 0

    .section .drectve, "yn"
    .ascii " -export:ec_data"

#--- sym-native.s
    .data
    .globl native_data
native_data:
    .word 0

    .section .drectve, "yn"
    .ascii " -export:native_data"

#--- ref-ec.s
    .data
    .rva __imp_ec_data

#--- ref-native.s
    .data
    .rva __imp_native_data
