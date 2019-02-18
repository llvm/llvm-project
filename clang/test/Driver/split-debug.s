// Check that we split debug output properly
//
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s
//
// CHECK-ACTIONS: "-split-dwarf-file" "split-debug.dwo"

// Check we pass -split-dwarf-file to `as` if -gsplit-dwarf=split.
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=split -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s

// Check we do not pass any -split-dwarf* commands to `as` if -gsplit-dwarf=single.
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf=single -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s

// RUN: %clang -target x86_64-macosx -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s
//
// CHECK-NO-ACTIONS-NOT: -split-dwarf


// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -o Bad.x -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-BAD < %t %s
//
// CHECK-BAD-NOT: "Bad.dwo"

