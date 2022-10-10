// Check that we verify debug output properly with multiple -arch options.
//
// RUN: %clang -target x86_64-apple-darwin11 -ccc-print-phases \
// RUN:   --verify-debug-info -arch i386 -arch x86_64 %s -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-MULTIARCH-ACTIONS < %t %s
//
// CHECK-MULTIARCH-ACTIONS: 0: input, "{{.*}}darwin-verify-debug.c", c
// CHECK-MULTIARCH-ACTIONS: 9: dsymutil, {8}, dSYM
// CHECK-MULTIARCH-ACTIONS: 10: verify-debug-info, {9}, none
//
// RUN: %clang -target x86_64-apple-darwin11 -ccc-print-bindings \
// RUN:   --verify-debug-info -arch i386 -arch x86_64 %s -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-MULTIARCH-BINDINGS < %t %s
//
// CHECK-MULTIARCH-BINDINGS: # "x86_64-apple-darwin11" - "darwin::Dsymutil", inputs: ["a.out"], output: "a.out.dSYM"
// CHECK-MULTIARCH-BINDINGS: # "x86_64-apple-darwin11" - "darwin::VerifyDebug", inputs: ["a.out.dSYM"], output: (nothing)

// Check output name derivation.
//
// RUN: %clang -target x86_64-apple-darwin11 -ccc-print-bindings \
// RUN:   --verify-debug-info -o foo %s -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-OUTPUT-NAME < %t %s
//
// CHECK-OUTPUT-NAME: "x86_64-apple-darwin11" - "darwin::Linker", inputs: [{{.*}}], output: "foo"
// CHECK-OUTPUT-NAME: "x86_64-apple-darwin11" - "darwin::Dsymutil", inputs: ["foo"], output: "foo.dSYM"
// CHECK-OUTPUT-NAME: "x86_64-apple-darwin11" - "darwin::VerifyDebug", inputs: ["foo.dSYM"], output: (nothing)

// Check that we only verify when needed.
//
// RUN: touch %t.o
// RUN: %clang -target x86_64-apple-darwin11 -ccc-print-bindings \
// RUN:   --verify-debug-info -o foo %t.o -g 2> %t
// RUN: not grep "Verify" %t
