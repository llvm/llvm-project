// RUN: %clang -target i386-apple-darwin10 -### -x assembler -c %s \
// RUN:   -no-integrated-as -static -dynamic 2>%t
// RUN: FileCheck -check-prefix=STATIC_AND_DYNAMIC-32 --input-file %t %s
//
// CHECK-STATIC_AND_DYNAMIC-32: as{{(.exe)?}}" "-arch" "i386" "-force_cpusubtype_ALL" "-static" "-o"

// RUN: %clang -target x86_64-apple-darwin10 -### -x assembler -c %s \
// RUN:   -no-integrated-as -static 2>%t
// RUN: FileCheck -check-prefix=STATIC-64 --input-file %t %s
//
// CHECK-STATIC-64: as{{(.exe)?}}" "-arch" "x86_64" "-force_cpusubtype_ALL" "-o"

// RUN: %clang -target x86_64-apple-darwin10 -### \
// RUN:   -arch armv6 -no-integrated-as -x assembler -c %s 2>%t
// RUN: FileCheck -check-prefix=ARMV6 --input-file %t %s
//
// CHECK-ARMV6: as{{(.exe)?}}" "-arch" "armv6" "-o"
