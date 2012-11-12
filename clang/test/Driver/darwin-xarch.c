// RUN: %clang -target x86_64-apple-darwin10 -### \
// RUN:   -arch i386 -Xarch_i386 -mmacosx-version-min=10.4 \
// RUN:   -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5 \
// RUN:   -c %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-COMPILE < %t %s
//
// CHECK-COMPILE: clang{{.*}}" "-cc1" "-triple" "i386-apple-macosx10.4.0" 
// CHECK-COMPILE: clang{{.*}}" "-cc1" "-triple" "x86_64-apple-macosx10.5.0"

// RUN: %clang -target x86_64-apple-darwin10 -### \
// RUN:   -arch i386 -Xarch_i386 -Wl,-some-linker-arg -filelist X 2> %t
// RUN: FileCheck --check-prefix=CHECK-LINK < %t %s
//
// CHECK-LINK: ld{{.*}} "-arch" "i386"{{.*}} "-some-linker-arg"

// RUN: %clang -target x86_64-apple-darwin10 -### \
// RUN:   -arch armv7 -Xarch_armv7 -Wl,-some-linker-arg -filelist X 2> %t
// RUN: FileCheck --check-prefix=CHECK-ARMV7-LINK < %t %s
//
// CHECK-ARMV7-LINK: ld{{.*}} "-arch" "armv7"{{.*}} "-some-linker-arg"
