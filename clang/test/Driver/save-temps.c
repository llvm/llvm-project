// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target

// RUN: %clang -target x86_64-apple-darwin -save-temps -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-o" "save-temps.i"
// CHECK: "-emit-llvm-uselists"
// CHECK: "-disable-llvm-passes"
// CHECK: "-o" "save-temps.bc"
// CHECK: "-o" "save-temps.s"
// CHECK: "-o" "save-temps.o"
// CHECK: "-o" "a.out"

// Check -save-temps=cwd which should work the same as -save-temps above

// RUN: %clang -target x86_64-apple-darwin -save-temps=cwd -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CWD
// CWD: "-o" "save-temps.i"
// CWD: "-emit-llvm-uselists"
// CWD: "-disable-llvm-passes"
// CWD: "-o" "save-temps.bc"
// CWD: "-o" "save-temps.s"
// CWD: "-o" "save-temps.o"
// CWD: "-o" "a.out"

// Check for a single clang cc1 invocation when NOT using -save-temps.
// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -S %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO-TEMPS
// NO-TEMPS: "-cc1"
// NO-TEMPS: "-S"
// NO-TEMPS: "-x" "c"

// RUN: %clang -target x86_64-apple-darwin -save-temps -arch i386 -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MULT-ARCH
// MULT-ARCH: "-o" "save-temps-i386.i"
// MULT-ARCH: "-o" "save-temps-i386.bc"
// MULT-ARCH: "-o" "save-temps-i386.s"
// MULT-ARCH: "-o" "save-temps-i386.o"
// MULT-ARCH: "-o" "a.out-i386"
// MULT-ARCH: "-o" "save-temps-x86_64.i"
// MULT-ARCH: "-o" "save-temps-x86_64.bc"
// MULT-ARCH: "-o" "save-temps-x86_64.s"
// MULT-ARCH: "-o" "save-temps-x86_64.o"
// MULT-ARCH: "-o" "a.out-x86_64"
// MULT-ARCH: lipo
// MULT-ARCH: "-create" "-output" "a.out" "a.out-i386" "a.out-x86_64"

// RUN: %clang -target x86_64-apple-darwin -save-temps=cwd -arch i386 -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MULT-ARCH-CWD
// MULT-ARCH-CWD: "-o" "save-temps-i386.i"
// MULT-ARCH-CWD: "-o" "save-temps-i386.bc"
// MULT-ARCH-CWD: "-o" "save-temps-i386.s"
// MULT-ARCH-CWD: "-o" "save-temps-i386.o"
// MULT-ARCH-CWD: "-o" "a.out-i386"
// MULT-ARCH-CWD: "-o" "save-temps-x86_64.i"
// MULT-ARCH-CWD: "-o" "save-temps-x86_64.bc"
// MULT-ARCH-CWD: "-o" "save-temps-x86_64.s"
// MULT-ARCH-CWD: "-o" "save-temps-x86_64.o"
// MULT-ARCH-CWD: "-o" "a.out-x86_64"
// MULT-ARCH-CWD: lipo
// MULT-ARCH-CWD: "-create" "-output" "a.out" "a.out-i386" "a.out-x86_64"

// Check that temp files are saved in the same directory as the output file
// regardless of whether -o is specified.

// RUN: %clang -target x86_64-apple-darwin -save-temps=obj -o obj/dir/a.out -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-OBJ
// CHECK-OBJ: "-o" "obj/dir{{/|\\\\}}save-temps.i"
// CHECK-OBJ: "-disable-llvm-passes"
// CHECK-OBJ: "-o" "obj/dir{{/|\\\\}}save-temps.bc"
// CHECK-OBJ: "-o" "obj/dir{{/|\\\\}}save-temps.s"
// CHECK-OBJ: "-o" "obj/dir{{/|\\\\}}save-temps.o"
// CHECK-OBJ: "-o" "obj/dir{{/|\\\\}}a.out"

// RUN: %clang -target x86_64-apple-darwin -save-temps=obj -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-OBJ-NOO
// CHECK-OBJ-NOO: "-o" "save-temps.i"
// CHECK-OBJ-NOO: "-disable-llvm-passes"
// CHECK-OBJ-NOO: "-o" "save-temps.bc"
// CHECK-OBJ-NOO: "-o" "save-temps.s"
// CHECK-OBJ-NOO: "-o" "save-temps.o"
// CHECK-OBJ-NOO: "-o" "a.out"

// RUN: %clang -target i386-unknown-freebsd -save-temps -g -c %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-SAVE-TEMPS
// CHECK-SAVE-TEMPS: "-cc1as"
// CHECK-SAVE-TEMPS: "-dwarf-version={{.}}"

// RUN: %clang --target=arm-arm-none-eabi -march=armv8-m.main -mcmse -save-temps -c -v %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-SAVE-TEMPS-CMSE
// RUN: %clang --target=arm-arm-none-eabi -march=armv8-m.main -mcmse -x assembler -c -v %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-SAVE-TEMPS-CMSE
// CHECK-SAVE-TEMPS-CMSE: -cc1as
// CHECK-SAVE-TEMPS-CMSE: +8msecext
// CHECK-SAVE-TEMPS-CMSE-NOT: '+cmse' is not a recognized feature for this target (ignoring feature)
