// Test AArch64 build attributes to ELF: 'aeabi_feature_and_bits'

// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=bti+pac-ret+gcs %s -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=ALL
// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=bti %s  -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=BTI
// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=pac-ret %s  -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=PAC
// RUN: %clang -c --target=aarch64-none-elf -mbranch-protection=gcs %s  -o %t
// RUN: llvm-readelf --hex-dump=.ARM.attributes %t | FileCheck %s -check-prefix=GCS

// ALL: Hex dump of section '.ARM.attributes':
// ALL-NEXT: 0x00000000 41230000 00616561 62695f66 65617475 A#...aeabi_featu
// ALL-NEXT: 0x00000010 72655f61 6e645f62 69747300 01000001 re_and_bits.....
// ALL-NEXT: 0x00000020 01010201

// BTI: Hex dump of section '.ARM.attributes':
// BTI-NEXT: 0x00000000 41230000 00616561 62695f66 65617475 A#...aeabi_featu
// BTI-NEXT: 0x00000010 72655f61 6e645f62 69747300 01000001 re_and_bits.....
// BTI-NEXT: 0x00000020 01000200

// PAC: Hex dump of section '.ARM.attributes':
// PAC-NEXT: 0x00000000 41230000 00616561 62695f66 65617475 A#...aeabi_featu
// PAC-NEXT: 0x00000010 72655f61 6e645f62 69747300 01000000 re_and_bits.....
// PAC-NEXT: 0x00000020 01010200

// GCS: Hex dump of section '.ARM.attributes':
// GCS-NEXT: 0x00000000 41230000 00616561 62695f66 65617475 A#...aeabi_featu
// GCS-NEXT: 0x00000010 72655f61 6e645f62 69747300 01000000 re_and_bits.....
// GCS-NEXT: 0x00000020 01000201