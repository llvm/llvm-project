// The following block tests:
//   - A compilation database is detected from build path specified by -p and
//     -include was provided.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/CompilationInc
// RUN: mkdir -p %T/CompilationInc
// RUN: mkdir -p %T/CompilationInc/a1
// RUN: mkdir -p %T/CompilationInc/a2
// RUN: mkdir -p %T/CompilationInc/a3

// This test uses a compilation database
// RUN: sed -e 's#$(path)#%/T/CompilationInc#g' %S/Inputs/compile_commands.json > %T/CompilationInc/compile_commands.json

// Check that files are transformed when -p and -include are specified.
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationInc/a1
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationInc/a2
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationInc/a3

// RUN: clang-modernize -use-nullptr -p=%T/CompilationInc -include=%T/CompilationInc/a1,%T/CompilationInc/a3
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/CompilationInc/a1/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/CompilationInc/a2/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/CompilationInc/a3/compilations.cpp
