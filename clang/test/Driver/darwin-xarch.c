// RUN: %clang --target=x86_64-apple-darwin10 -### \
// RUN:   -arch i386 -Xarch_i386 -mmacosx-version-min=10.4 \
// RUN:   -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5 \
// RUN:   -c %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-COMPILE < %t %s

// CHECK-COMPILE: "-cc1" "-triple" "i386-apple-macosx10.4.0" 
// CHECK-COMPILE: "-cc1" "-triple" "x86_64-apple-macosx10.5.0"

// RUN: %clang --target=x86_64-apple-darwin10 -### \
// RUN:   -arch i386 -Xarch_i386 -Wl,-some-linker-arg -filelist X 2> %t
// RUN: FileCheck --check-prefix=CHECK-LINK < %t %s

// CHECK-LINK: ld{{.*}} "-arch" "i386"{{.*}} "-some-linker-arg"

// RUN: %clang --target=x86_64-apple-darwin10 -### \
// RUN:   -arch armv7 -Xarch_armv7 -Wl,-some-linker-arg -filelist X 2> %t
// RUN: FileCheck --check-prefix=CHECK-ARMV7-LINK < %t %s

// CHECK-ARMV7-LINK: ld{{.*}} "-arch" "armv7"{{.*}} "-some-linker-arg"


// RUN: %clang --target=armv7s-apple-ios7 -### \
// RUN:   -arch armv7  -Xarch_armv7  -DARMV7=1 \
// RUN:   -arch armv7s -Xarch_armv7s -DARMV7S=1 \
// RUN:   -c %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ARMV7S < %t %s

// CHECK-ARMV7S: "-cc1" "-triple" "thumbv7-apple-ios7.0.0"
// CHECK-ARMV7S-NOT:  "-D" "ARMV7S=1"
// CHECK-ARMV7S-SAME: "-D" "ARMV7=1"

// CHECK-ARMV7S: "-cc1" "-triple" "thumbv7s-apple-ios7.0.0"
// CHECK-ARMV7S-NOT:  "-D" "ARMV7=1"
// CHECK-ARMV7S-SAME: "-D" "ARMV7S=1"

// RUN: %clang --target=arm64-apple-ios14 -### \
// RUN:   -arch arm64  -Xarch_arm64  -DARM64=1 \
// RUN:   -arch arm64e -Xarch_arm64e -DARM64E=1 \
// RUN:   -c %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ARM64 < %t %s

// CHECK-ARM64: "-cc1" "-triple" "arm64-apple-ios14.0.0"
// CHECK-ARM64-NOT:  "-D" "ARM64E=1"
// CHECK-ARM64-SAME: "-D" "ARM64=1"

// CHECK-ARM64: "-cc1" "-triple" "arm64e-apple-ios14.0.0"
// CHECK-ARM64-NOT:  "-D" "ARM64=1"
// CHECK-ARM64-SAME: "-D" "ARM64E=1"
