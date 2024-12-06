// Test AArch64 build attributes to ELF: 'aeabi-feature-and-bits'

// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=bti+pac-ret+gcs %s -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=ALL
// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=bti %s  -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=BTI
// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=pac-ret %s  -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=PAC
// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=gcs %s  -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=GCS

// ALL: Hex dump of section '.ARM.attributes':
// ALL-NEXT: 0x00000000 41230000 00616561 62692d66 65617475 A#...aeabi-featu
// ALL-NEXT: 0x00000010 72652d61 6e642d62 69747301 00000101 re-and-bits.....
// ALL-NEXT: 0x00000020 010201

// BTI: Hex dump of section '.ARM.attributes':
// BTI-NEXT: 0x00000000 41230000 00616561 62692d66 65617475 A#...aeabi-featu
// BTI-NEXT: 0x00000010 72652d61 6e642d62 69747301 00000101 re-and-bits.....
// BTI-NEXT: 0x00000020 000200

// PAC: Hex dump of section '.ARM.attributes':
// PAC-NEXT: 0x00000000 41230000 00616561 62692d66 65617475 A#...aeabi-featu
// PAC-NEXT: 0x00000010 72652d61 6e642d62 69747301 00000001 re-and-bits.....
// PAC-NEXT: 0x00000020 010200

// GCS: Hex dump of section '.ARM.attributes':
// GCS-NEXT: 0x00000000 41230000 00616561 62692d66 65617475 A#...aeabi-featu
// GCS-NEXT: 0x00000010 72652d61 6e642d62 69747301 00000001 re-and-bits.....
// GCS-NEXT: 0x00000020 000201