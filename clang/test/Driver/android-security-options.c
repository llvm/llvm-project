// Check that Android enables PAC and BTI by default on AArch64.
// RUN: %clang --target=aarch64-linux-android -### -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ANDROID-DEFAULT
// ANDROID-DEFAULT: "-msign-return-address=non-leaf" "-msign-return-address-key=a_key" "-mbranch-target-enforce"

// Check that the Android default can be overridden.
// RUN: %clang --target=aarch64-linux-android -mbranch-protection=none -### -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ANDROID-OVERRIDE
// ANDROID-OVERRIDE: "-msign-return-address=none"
// ANDROID-OVERRIDE-NOT: "-msign-return-address-key"
// ANDROID-OVERRIDE-NOT: "-mbranch-target-enforce"

// Check that Android enables BTI by default when -msign-return-address is passed.
// RUN: %clang --target=aarch64-linux-android -msign-return-address=non-leaf -### -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ANDROID-PAC-BTI
// ANDROID-PAC-BTI: "-msign-return-address=non-leaf" "-msign-return-address-key=a_key" "-mbranch-target-enforce"