// Basic compilation for various types of files.
// RUN: %clang -target i386-unknown-unknown -ccc-print-phases -x c %s -x objective-c %s -x c++ %s -x objective-c++ -x assembler %s -x assembler-with-cpp %s -x none %s 2>&1 | FileCheck -check-prefix=BASIC %s
// BASIC: 0: input, "{{.*}}phases.c", c
// BASIC: 1: preprocessor, {0}, cpp-output
// BASIC: 2: compiler, {1}, assembler
// BASIC: 3: assembler, {2}, object
// BASIC: 4: input, "{{.*}}phases.c", objective-c
// BASIC: 5: preprocessor, {4}, objective-c-cpp-output
// BASIC: 6: compiler, {5}, assembler
// BASIC: 7: assembler, {6}, object
// BASIC: 8: input, "{{.*}}phases.c", c++
// BASIC: 9: preprocessor, {8}, c++-cpp-output
// BASIC: 10: compiler, {9}, assembler
// BASIC: 11: assembler, {10}, object
// BASIC: 12: input, "{{.*}}phases.c", assembler
// BASIC: 13: assembler, {12}, object
// BASIC: 14: input, "{{.*}}phases.c", assembler-with-cpp
// BASIC: 15: preprocessor, {14}, assembler
// BASIC: 16: assembler, {15}, object
// BASIC: 17: input, "{{.*}}phases.c", c
// BASIC: 18: preprocessor, {17}, cpp-output
// BASIC: 19: compiler, {18}, assembler
// BASIC: 20: assembler, {19}, object
// BASIC: 21: linker, {3, 7, 11, 13, 16, 20}, image

// Universal linked image.
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -x c %s -arch ppc -arch i386 2>&1 | FileCheck -check-prefix=ULI %s
// ULI: 0: input, "{{.*}}phases.c", c
// ULI: 1: preprocessor, {0}, cpp-output
// ULI: 2: compiler, {1}, assembler
// ULI: 3: assembler, {2}, object
// ULI: 4: linker, {3}, image
// ULI: 5: bind-arch, "ppc", {4}, image
// ULI: 6: bind-arch, "i386", {4}, image
// ULI: 7: lipo, {5, 6}, image

// Universal object file.
// RUN: %clang -target i386-apple-darwin9 -ccc-print-phases -c -x c %s -arch ppc -arch i386 2>&1 | FileCheck -check-prefix=UOF %s
// UOF: 0: input, "{{.*}}phases.c", c
// UOF: 1: preprocessor, {0}, cpp-output
// UOF: 2: compiler, {1}, assembler
// UOF: 3: assembler, {2}, object
// UOF: 4: bind-arch, "ppc", {3}, object
// UOF: 5: bind-arch, "i386", {3}, object
// UOF: 6: lipo, {4, 5}, object

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

