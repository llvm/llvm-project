// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows aarch64.s -o aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows arm64ec.s -o arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows tls-aarch64.s -o tls-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows tls-arm64ec.s -o tls-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj

// RUN: rm -f tls.lib
// RUN: llvm-lib -machine:arm64x -out:tls.lib tls-aarch64.obj tls-arm64ec.obj loadconfig-arm64ec.obj loadconfig-arm64.obj
// RUN: lld-link -machine:arm64x -dll -noentry aarch64.obj arm64ec.obj tls.lib -out:out.dll

// Check that we're using the right _tls_index and _tls_used for both views.

// RUN: llvm-readobj --coff-tls-directory --hex-dump=.ec --hex-dump=.a64 out.dll | FileCheck %s
// CHECK:      Format: COFF-ARM64X
// CHECK-NEXT: Arch: aarch64
// CHECK-NEXT: AddressSize: 64bit
// CHECK-EMPTY:
// CHECK-NEXT: Hex dump of section '.a64':
// CHECK-NEXT: 0x180004000 00500000 04500000 00800000 01800000
// CHECK-EMPTY:
// CHECK-NEXT: Hex dump of section '.ec':
// CHECK-NEXT: 0x180007000 00600000 04600000 00800000 01800000
// CHECK-NEXT: TLSDirectory {
// CHECK-NEXT:   StartAddressOfRawData: 0x180008000
// CHECK-NEXT:   EndAddressOfRawData: 0x180008001
// CHECK-NEXT:   AddressOfIndex: 0x180005000
// CHECK-NEXT:   AddressOfCallBacks: 0x0
// CHECK-NEXT:   SizeOfZeroFill: 0x0
// CHECK-NEXT:   Characteristics [ (0x100000)
// CHECK-NEXT:     IMAGE_SCN_ALIGN_1BYTES (0x100000)
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: HybridObject {
// CHECK-NEXT:   Format: COFF-ARM64EC
// CHECK-NEXT:   Arch: aarch64
// CHECK-NEXT:   AddressSize: 64bit
// CHECK-EMPTY:
// CHECK-NEXT:   Hex dump of section '.a64':
// CHECK-NEXT:   0x180004000 00500000 04500000 00800000 01800000
// CHECK-EMPTY:
// CHECK-NEXT:   Hex dump of section '.ec':
// CHECK-NEXT:   0x180007000 00600000 04600000 00800000 01800000
// CHECK-NEXT:   TLSDirectory {
// CHECK-NEXT:     StartAddressOfRawData: 0x180008000
// CHECK-NEXT:     EndAddressOfRawData: 0x180008001
// CHECK-NEXT:     AddressOfIndex: 0x180006000
// CHECK-NEXT:     AddressOfCallBacks: 0x0
// CHECK-NEXT:     SizeOfZeroFill: 0x0
// CHECK-NEXT:     Characteristics [ (0x100000)
// CHECK-NEXT:       IMAGE_SCN_ALIGN_1BYTES (0x100000)
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// RUN: lld-link -machine:arm64x -dll -noentry arm64ec.obj tls.lib -out:out-ec.dll
// RUN: llvm-readobj --coff-tls-directory out-ec.dll | FileCheck --check-prefix=EC-ONLY %s
// EC-ONLY:      Format: COFF-ARM64X
// EC-ONLY-NEXT: Arch: aarch64
// EC-ONLY-NEXT: AddressSize: 64bit
// EC-ONLY-NEXT: TLSDirectory {
// EC-ONLY-NEXT: }
// EC-ONLY-NEXT: HybridObject {
// EC-ONLY-NEXT:   Format: COFF-ARM64EC
// EC-ONLY-NEXT:   Arch: aarch64
// EC-ONLY-NEXT:   AddressSize: 64bit
// EC-ONLY-NEXT:   TLSDirectory {
// EC-ONLY-NEXT:     StartAddressOfRawData: 0x180006000
// EC-ONLY-NEXT:     EndAddressOfRawData: 0x180006001
// EC-ONLY-NEXT:     AddressOfIndex: 0x180004000
// EC-ONLY-NEXT:     AddressOfCallBacks: 0x0
// EC-ONLY-NEXT:     SizeOfZeroFill: 0x0
// EC-ONLY-NEXT:     Characteristics [ (0x100000)
// EC-ONLY-NEXT:       IMAGE_SCN_ALIGN_1BYTES (0x100000)
// EC-ONLY-NEXT:     ]
// EC-ONLY-NEXT:   }
// EC-ONLY-NEXT: }

// RUN: lld-link -machine:arm64x -dll -noentry aarch64.obj tls.lib -out:out-native.dll
// RUN: llvm-readobj --coff-tls-directory out-native.dll | FileCheck --check-prefix=NATIVE-ONLY %s
// NATIVE-ONLY:      Format: COFF-ARM64X
// NATIVE-ONLY-NEXT: Arch: aarch64
// NATIVE-ONLY-NEXT: AddressSize: 64bit
// NATIVE-ONLY-NEXT: TLSDirectory {
// NATIVE-ONLY-NEXT:   StartAddressOfRawData: 0x180006000
// NATIVE-ONLY-NEXT:   EndAddressOfRawData: 0x180006001
// NATIVE-ONLY-NEXT:   AddressOfIndex: 0x180005000
// NATIVE-ONLY-NEXT:   AddressOfCallBacks: 0x0
// NATIVE-ONLY-NEXT:   SizeOfZeroFill: 0x0
// NATIVE-ONLY-NEXT:   Characteristics [ (0x100000)
// NATIVE-ONLY-NEXT:     IMAGE_SCN_ALIGN_1BYTES (0x100000)
// NATIVE-ONLY-NEXT:   ]
// NATIVE-ONLY-NEXT: }
// NATIVE-ONLY-NEXT: HybridObject {
// NATIVE-ONLY-NEXT:   Format: COFF-ARM64EC
// NATIVE-ONLY-NEXT:   Arch: aarch64
// NATIVE-ONLY-NEXT:   AddressSize: 64bit
// NATIVE-ONLY-NEXT:   TLSDirectory {
// NATIVE-ONLY-NEXT:   }
// NATIVE-ONLY-NEXT: }

#--- tls-aarch64.s
        .section .defa64,"dr",discard,_tls_index
        .globl _tls_index
_tls_index:
        .long 0

        .section .tls,"dr",discard,_tls_start
        .globl _tls_start
_tls_start:
        .byte 0

        .section .tls$ZZZ,"dr",discard,_tls_end
        .globl _tls_end
_tls_end:
        .byte 0

        .section .defa64,"dr",discard,_tls_used
        .globl _tls_used
_tls_used:
        .xword _tls_start
        .xword _tls_end
        .xword _tls_index
        .xword 0
        .xword 0

#--- aarch64.s
        .section .a64,"dr"
        .rva _tls_index
        .rva _tls_used
        .rva _tls_start
        .rva _tls_end

#--- tls-arm64ec.s
        .section .defec,"dr",discard,_tls_index
        .globl _tls_index
_tls_index:
        .long 0

        .section .tls,"dr",discard,_tls_start
        .globl _tls_start
_tls_start:
        .byte 0

        .section .tls$ZZZ,"dr",discard,_tls_end
        .globl _tls_end
_tls_end:
        .byte 0

        .section .defec,"dr",discard,_tls_used
        .globl _tls_used
_tls_used:
        .xword _tls_start
        .xword _tls_end
        .xword _tls_index
        .xword 0
        .xword 0

#--- arm64ec.s
        .section .ec,"dr"
        .rva _tls_index
        .rva _tls_used
        .rva _tls_start
        .rva _tls_end
