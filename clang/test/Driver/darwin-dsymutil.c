// Check that we run dsymutil properly with multiple -arch options.

// RUN: %clang -target x86_64-apple-darwin10 -ccc-print-phases \
// RUN:   -arch i386 -arch x86_64 %s -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-MULTIARCH-ACTIONS < %t %s

// CHECK-MULTIARCH-ACTIONS: 0: input, "{{.*}}darwin-dsymutil.c", c
// CHECK-MULTIARCH-ACTIONS: 1: preprocessor, {0}, cpp-output
// CHECK-MULTIARCH-ACTIONS: 2: compiler, {1}, ir
// CHECK-MULTIARCH-ACTIONS: 3: backend, {2}, assembler
// CHECK-MULTIARCH-ACTIONS: 4: assembler, {3}, object
// CHECK-MULTIARCH-ACTIONS: 5: linker, {4}, image
// CHECK-MULTIARCH-ACTIONS: 6: bind-arch, "i386", {5}, image
// CHECK-MULTIARCH-ACTIONS: 7: bind-arch, "x86_64", {5}, image
// CHECK-MULTIARCH-ACTIONS: 8: lipo, {6, 7}, image
// CHECK-MULTIARCH-ACTIONS: 9: dsymutil, {8}, dSYM

// RUN: %clang -target x86_64-apple-darwin10 -ccc-print-bindings \
// RUN:   -arch i386 -arch x86_64 %s -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-MULTIARCH-BINDINGS < %t %s

// CHECK-MULTIARCH-BINDINGS: "x86_64-apple-darwin10" - "darwin::Lipo", inputs: [{{.*}}, {{.*}}], output: "a.out"
// CHECK-MULTIARCH-BINDINGS: # "x86_64-apple-darwin10" - "darwin::Dsymutil", inputs: ["a.out"], output: "a.out.dSYM"

// Check output name derivation.

// RUN: %clang -target x86_64-apple-darwin10 -ccc-print-bindings \
// RUN:   -o foo %s -g 2> %t
// RUN: FileCheck -Doutfile=foo -Ddsymfile=foo.dSYM \
// RUN:          -check-prefix=CHECK-OUTPUT-NAME < %t %s

// RUN: %clang -target x86_64-apple-darwin10 -ccc-print-bindings \
// RUN:   -o bar/foo %s -g 2> %t
// RUN: FileCheck -Doutfile=bar/foo -Ddsymfile=bar/foo.dSYM \
// RUN:           -check-prefix=CHECK-OUTPUT-NAME < %t %s

// RUN: %clang -target x86_64-apple-darwin10 -ccc-print-bindings \
// RUN:   -o bar/foo -dsym-dir external %s -g 2> %t
// RUN: FileCheck -Doutfile=bar/foo -Ddsymfile=external/foo.dSYM \
// RUN:           -check-prefix=CHECK-OUTPUT-NAME < %t %s

// CHECK-OUTPUT-NAME: "x86_64-apple-darwin10" - "darwin::Linker", inputs: [{{.*}}], output: "[[outfile]]"
// CHECK-OUTPUT-NAME: "x86_64-apple-darwin10" - "darwin::Dsymutil", inputs: ["[[outfile]]"], output: "[[dsymfile]]"

// Check output name derivation for multiple -arch options.

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -arch x86_64 -arch arm64 -ccc-print-bindings %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MULTIARCH-OUTPUT-NAME < %t %s

// CHECK-MULTIARCH-OUTPUT-NAME: "x86_64-apple-darwin10" - "darwin::Linker", inputs: ["{{.*}}{{/|\\}}darwin-dsymutil-x86_64.o"], output: "{{.*}}{{/|\\}}darwin-dsymutil-x86_64.out"
// CHECK-MULTIARCH-OUTPUT-NAME: "arm64-apple-darwin10" - "darwin::Linker", inputs: ["{{.*}}{{/|\\}}darwin-dsymutil-arm64.o"], output: "{{.*}}{{/|\\}}darwin-dsymutil-arm64.out"
// CHECK-MULTIARCH-OUTPUT-NAME: "arm64-apple-darwin10" - "darwin::Lipo", inputs: ["{{.*}}{{/|\\}}darwin-dsymutil-x86_64.out", "{{.*}}{{/|\\}}darwin-dsymutil-arm64.out"], output: "a.out"

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -Wl,-foo -arch x86_64 -arch arm64 -ccc-print-bindings %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MULTIARCH-OUTPUT-NAME-WITH-ARG < %t %s

// CHECK-MULTIARCH-OUTPUT-NAME-WITH-ARG: "x86_64-apple-darwin10" - "darwin::Linker", inputs: [(input arg), "{{.*}}{{/|\\}}darwin-dsymutil-x86_64.o"], output: "{{.*}}{{/|\\}}darwin-dsymutil-x86_64.out"
// CHECK-MULTIARCH-OUTPUT-NAME-WITH-ARG: "arm64-apple-darwin10" - "darwin::Linker", inputs: [(input arg), "{{.*}}{{/|\\}}darwin-dsymutil-arm64.o"], output: "{{.*}}{{/|\\}}darwin-dsymutil-arm64.out"
// CHECK-MULTIARCH-OUTPUT-NAME-WITH-ARG: "arm64-apple-darwin10" - "darwin::Lipo", inputs: ["{{.*}}{{/|\\}}darwin-dsymutil-x86_64.out", "{{.*}}{{/|\\}}darwin-dsymutil-arm64.out"], output: "a.out"

// Check that we only use dsymutil when needed.

// RUN: touch %t.o
// RUN: %clang -target x86_64-apple-darwin10 -ccc-print-bindings \
// RUN:   -o foo %t.o -g 2> %t
// RUN: not grep "Dsymutil" %t

// Check that we don't crash when translating arguments for dsymutil.
// RUN: %clang -m32 -target x86_64-apple-darwin10 -arch x86_64 -g %s -###
