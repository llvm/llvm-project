// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -mbranch-protection=bti \
// RUN: -mno-bti-at-return-twice -### %s 2>&1 | FileCheck %s --check-prefix=FEAT
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -mbranch-protection=bti \
// RUN: -### %s 2>&1 | FileCheck %s --check-prefix=NOFEAT

// FEAT: "+no-bti-at-return-twice"
// NOFEAT-NOT: "+no-bti-at-return-twice"
