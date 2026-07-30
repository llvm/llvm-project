// Basic compilation for various types of files.
// RUN: %clang -target i386-unknown-unknown -ccc-print-phases -x c %s -x objective-c %s -x c++ %s -x objective-c++ -x assembler %s -x assembler-with-cpp %s -x none %s 2>&1 | FileCheck -check-prefix=BASIC %s
// BASIC: 0: input, "{{.*}}phases.c", c
// BASIC: 1: preprocessor, {0}, cpp-output
// BASIC: 2: compiler, {1}, ir
// BASIC: 3: backend, {2}, assembler
// BASIC: 4: assembler, {3}, object
// BASIC: 5: input, "{{.*}}phases.c", objective-c
// BASIC: 6: preprocessor, {5}, objective-c-cpp-output
// BASIC: 7: compiler, {6}, ir
// BASIC: 8: backend, {7}, assembler
// BASIC: 9: assembler, {8}, object
// BASIC: 10: input, "{{.*}}phases.c", c++
// BASIC: 11: preprocessor, {10}, c++-cpp-output
// BASIC: 12: compiler, {11}, ir
// BASIC: 13: backend, {12}, assembler
// BASIC: 14: assembler, {13}, object
// BASIC: 15: input, "{{.*}}phases.c", assembler
// BASIC: 16: assembler, {15}, object
// BASIC: 17: input, "{{.*}}phases.c", assembler-with-cpp
// BASIC: 18: preprocessor, {17}, assembler
// BASIC: 19: assembler, {18}, object
// BASIC: 20: input, "{{.*}}phases.c", c
// BASIC: 21: preprocessor, {20}, cpp-output
// BASIC: 22: compiler, {21}, ir
// BASIC: 23: backend, {22}, assembler
// BASIC: 24: assembler, {23}, object
// BASIC: 25: linker, {4, 9, 14, 16, 19, 24}, image

// Universal linked image.
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -x c %s -arch arm -arch i386 2>&1 | FileCheck -check-prefix=ULI %s
// ULI: 0: input, "{{.*}}phases.c", c
// ULI: 1: preprocessor, {0}, cpp-output
// ULI: 2: compiler, {1}, ir
// ULI: 3: backend, {2}, assembler
// ULI: 4: assembler, {3}, object
// ULI: 5: linker, {4}, image
// ULI: 6: bind-arch, "arm", {5}, image
// ULI: 7: bind-arch, "i386", {5}, image
// ULI: 8: lipo, {6, 7}, image

// Universal object file.
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -c -x c %s -arch arm -arch i386 2>&1 | FileCheck -check-prefix=UOF %s
// UOF: 0: input, "{{.*}}phases.c", c
// UOF: 1: preprocessor, {0}, cpp-output
// UOF: 2: compiler, {1}, ir
// UOF: 3: backend, {2}, assembler
// UOF: 4: assembler, {3}, object
// UOF: 5: bind-arch, "arm", {4}, object
// UOF: 6: bind-arch, "i386", {4}, object
// UOF: 7: lipo, {5, 6}, object

// Arch defaulting
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -c -x assembler %s 2>&1 | FileCheck -check-prefix=ARCH1 %s
// ARCH1: 2: bind-arch, "i386", {1}, object
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -c -x assembler %s -m32 -m64 2>&1 | FileCheck -check-prefix=ARCH2 %s
// ARCH2: 2: bind-arch, "x86_64", {1}, object
// RUN: %clang -target x86_64-apple-darwin9 -ccc-print-phases -c -x assembler %s 2>&1 | FileCheck -check-prefix=ARCH3 %s
// ARCH3: 2: bind-arch, "x86_64", {1}, object
// RUN: %clang -target x86_64-apple-darwin9 -ccc-print-phases -c -x assembler %s -m64 -m32 2>&1 | FileCheck -check-prefix=ARCH4 %s
// ARCH4: 2: bind-arch, "i386", {1}, object

// Analyzer
// RUN: %clang -target i386-unknown-unknown -ccc-print-phases --analyze %s 2>&1 | FileCheck -check-prefix=ANALYZE %s
// ANALYZE: 0: input, "{{.*}}phases.c", c
// ANALYZE: 1: preprocessor, {0}, cpp-output
// ANALYZE: 2: analyzer, {1}, plist

// Precompiler
// RUN: %clang -target i386-unknown-unknown -ccc-print-phases -x c-header %s 2>&1 | FileCheck -check-prefix=PCH %s
// PCH: 0: input, "{{.*}}phases.c", c-header
// PCH: 1: preprocessor, {0}, c-header-cpp-output
// PCH: 2: precompiler, {1}, precompiled-header

// Darwin overrides the handling for .s
// RUN: touch %t.s
// RUN: %clang -target i386-unknown-unknown -ccc-print-phases -c %t.s 2>&1 | FileCheck -check-prefix=DARWIN1 %s
// DARWIN1: 0: input, "{{.*}}.s", assembler
// DARWIN1: 1: assembler, {0}, object
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -c %t.s 2>&1 | FileCheck -check-prefix=DARWIN2 %s
// DARWIN2: 0: input, "{{.*}}.s", assembler-with-cpp
// DARWIN2: 1: preprocessor, {0}, assembler
// DARWIN2: 2: assembler, {1}, object

