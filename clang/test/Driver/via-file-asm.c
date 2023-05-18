// Should save and read back the assembly from a file
// RUN: %clang -target arm-none-linux-gnueabi -integrated-as -via-file-asm %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK: "-o" "[[TMP:[^"]*]]"
// CHECK: -cc1as
// CHECK: [[TMP]]

// Should not force using the integrated assembler
// RUN: %clang -target arm-none-linux-gnueabi -no-integrated-as -via-file-asm %s -### 2>&1 | FileCheck --check-prefix=NO_IAS %s
// NO_IAS-NOT: "-cc1as"
