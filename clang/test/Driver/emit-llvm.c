// Check that -O4 is only honored as the effective -O option.
// <rdar://problem/7046672> clang/loader problem

// RUN: %clang -ccc-print-phases -c -O4 -O0 %s 2> %t
// RUN: FileCheck --check-prefix=O4_AND_O0 %s < %t

// O4_AND_O0: 0: input, "{{.*}}", c
// O4_AND_O0: 1: preprocessor, {0}, cpp-output
// O4_AND_O0: 2: compiler, {1}, assembler
// O4_AND_O0: 3: assembler, {2}, object
