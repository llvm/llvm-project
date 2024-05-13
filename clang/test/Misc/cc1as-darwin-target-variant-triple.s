// Run cc1as using darwin-target-variant-triple
// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64-apple-macos10.9 -darwin-target-variant-triple x86_64-apple-ios13.1-macabi -filetype obj %s -o - \
// RUN: | llvm-readobj --file-headers --macho-version-min - \
// RUN: | FileCheck --check-prefix=CHECK %s

// CHECK: File: <stdin>
// CHECK-NEXT: Format: Mach-O 64-bit x86-64
// CHECK-NEXT: Arch: x86_64
// CHECK-NEXT: AddressSize: 64bit
// CHECK-NEXT: MachHeader {
// CHECK-NEXT:   Magic: Magic64 (0xFEEDFACF)
// CHECK-NEXT:   CpuType: X86-64 (0x1000007)
// CHECK-NEXT:   CpuSubType: CPU_SUBTYPE_X86_64_ALL (0x3)
// CHECK-NEXT:   FileType: Relocatable (0x1)
// CHECK-NEXT:   NumOfLoadCommands: 3
// CHECK-NEXT:   SizeOfLoadCommands: 192
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Reserved: 0x0
// CHECK-NEXT: }
// CHECK-NEXT: MinVersion {
// CHECK-NEXT:   Cmd: LC_VERSION_MIN_MACOSX
// CHECK-NEXT:   Size: 16
// CHECK-NEXT:   Version: 10.9
// CHECK-NEXT:   SDK: n/a
// CHECK-NEXT: }
// CHECK-NEXT: MinVersion {
// CHECK-NEXT:   Cmd: LC_BUILD_VERSION
// CHECK-NEXT:   Size: 24
// CHECK-NEXT:   Platform: macCatalyst
// CHECK-NEXT:   Version: 13.1
// CHECK-NEXT:   SDK: n/a
// CHECK-NEXT: }
