// Test AArch64 build attributes to assmebly: 'aeabi-feature-and-bits'

// RUN: %clang --target=aarch64-none-elf -mbranch-protection=bti+pac-ret+gcs -S -o- %s | FileCheck %s -check-prefix=ALL
// RUN: %clang --target=aarch64-none-elf -mbranch-protection=bti -S -o- %s | FileCheck %s -check-prefix=BTI
// RUN: %clang --target=aarch64-none-elf -mbranch-protection=pac-ret -S -o- %s | FileCheck %s -check-prefix=PAC
// RUN: %clang --target=aarch64-none-elf -mbranch-protection=gcs -S -o- %s | FileCheck %s -check-prefix=GCS

// ALL: .text
// ALL-NEXT: .aeabi_subsection .aeabi-feature-and-bits, optional, ULEB128
// ALL-NEXT: .aeabi_attribute Tag_Feature_BTI, 1
// ALL-NEXT: .aeabi_attribute Tag_Feature_PAC, 1
// ALL-NEXT: .aeabi_attribute Tag_Feature_GCS, 1

// BTI: .text
// BTI-NEXT: .aeabi_subsection .aeabi-feature-and-bits, optional, ULEB128
// BTI-NEXT: .aeabi_attribute Tag_Feature_BTI, 1
// BTI-NEXT: .aeabi_attribute Tag_Feature_PAC, 0
// BTI-NEXT: .aeabi_attribute Tag_Feature_GCS, 0

// PAC: .text
// PAC-NEXT: .aeabi_subsection .aeabi-feature-and-bits, optional, ULEB128
// PAC-NEXT: .aeabi_attribute Tag_Feature_BTI, 0
// PAC-NEXT: .aeabi_attribute Tag_Feature_PAC, 1
// PAC-NEXT: .aeabi_attribute Tag_Feature_GCS, 0

// GCS: .text
// GCS-NEXT: .aeabi_subsection .aeabi-feature-and-bits, optional, ULEB128
// GCS-NEXT: .aeabi_attribute Tag_Feature_BTI, 0
// GCS-NEXT: .aeabi_attribute Tag_Feature_PAC, 0
// GCS-NEXT: .aeabi_attribute Tag_Feature_GCS, 1