// RUN: %clang -### -target arm64-apple-macosx11 -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -target arm64-apple-macosx11 -fobjc-constant-literals -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -target arm64-apple-macosx11 -fno-objc-constant-literals -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLED

// DEFAULT-NOT: -fobjc-constant-literals
// DEFAULT-NOT: -fconstant-nsnumber-literals
// DEFAULT-NOT: -fconstant-nsarray-literals
// DEFAULT-NOT: -fconstant-nsdictionary-literals

// ENABLED: -fobjc-constant-literals
// ENABLED: -fconstant-nsnumber-literals
// ENABLED: -fconstant-nsarray-literals
// ENABLED: -fconstant-nsdictionary-literals

// DISABLED-NOT: -fobjc-constant-literals
// DISABLED-NOT: -fconstant-nsnumber-literals
// DISABLED-NOT: -fconstant-nsarray-literals
// DISABLED-NOT: -fconstant-nsdictionary-literals
