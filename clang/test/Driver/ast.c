// RUN: %clang -target i386-unknown-unknown -ccc-print-phases -emit-ast %s 2> %t
// RUN: echo 'END' >> %t
// RUN: FileCheck -check-prefix EMIT-AST-PHASES -input-file %t %s

// EMIT-AST-PHASES: 0: input,
// EMIT-AST-PHASES: , c
// EMIT-AST-PHASES: 1: preprocessor, {0}, cpp-output
// EMIT-AST-PHASES: 2: compiler, {1}, ast
// EMIT-AST-PHASES-NOT: 3:
// EMIT-AST-PHASES: END

// RUN: touch %t.ast
// RUN: %clang -target i386-unknown-unknown -ccc-print-phases -c %t.ast 2> %t
// RUN: echo 'END' >> %t
// RUN: FileCheck -check-prefix COMPILE-AST-PHASES -input-file %t %s

// COMPILE-AST-PHASES: 0: input,
// COMPILE-AST-PHASES: , ast
// COMPILE-AST-PHASES: 1: compiler, {0}, assembler
// COMPILE-AST-PHASES: 2: assembler, {1}, object
// COMPILE-AST-PHASES-NOT: 3:
// COMPILE-AST-PHASES: END

// FIXME: There is a problem with compiling AST's in that the input language is
// not available for use by other tools (for example, to automatically add
// -lstdc++). We may need -x [objective-]c++-ast and all that goodness. :(
