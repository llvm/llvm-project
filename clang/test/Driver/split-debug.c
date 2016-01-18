// Check that we split debug output properly
//
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s
//
// CHECK-ACTIONS: objcopy{{.*}}--extract-dwo{{.*}}"split-debug.dwo"
// CHECK-ACTIONS: objcopy{{.*}}--strip-dwo{{.*}}"split-debug.o"


// RUN: %clang -target x86_64-macosx -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s
//
// CHECK-NO-ACTIONS-NOT: -split-dwarf


// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -o Bad.x -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-BAD < %t %s
//
// CHECK-BAD-NOT: "Bad.dwo"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-OPTION < %t %s
//
// CHECK-OPTION: "-split-dwarf-file" "split-debug.dwo"

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ASM < %t %s
//
// CHECK-ASM-NOT: objcopy

// RUN: %clang -target x86_64-unknown-linux-gnu -no-integrated-as -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-IAS < %t %s
//
// CHECK-IAS: objcopy

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -gmlt -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-GMLT-OVER-SPLIT < %t %s
//
// CHECK-GMLT-OVER-SPLIT: "-debug-info-kind=line-tables-only"
// CHECK-GMLT-OVER-SPLIT-NOT: "-split-dwarf=Enable"
// CHECK-GMLT-OVER-SPLIT-NOT: "-split-dwarf-file"

// RUN: %clang -target x86_64-unknown-linux-gnu -gmlt -gsplit-dwarf -S -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-SPLIT-OVER-GMLT < %t %s
//
// CHECK-SPLIT-OVER-GMLT: "-split-dwarf=Enable" "-debug-info-kind=limited"
// CHECK-SPLIT-OVER-GMLT: "-split-dwarf-file"
