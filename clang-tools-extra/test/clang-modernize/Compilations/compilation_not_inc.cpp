// The following block tests:
//   - A compilation database is detected from build path specified by -p but
//     neither sources nor -include was provided.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/CompilationNotInc
// RUN: mkdir -p %T/CompilationNotInc
// RUN: mkdir -p %T/CompilationNotInc/a1
// RUN: mkdir -p %T/CompilationNotInc/a2
// RUN: mkdir -p %T/CompilationNotInc/a3

// This test uses a compilation database
// RUN: sed -e 's#$(path)#%/T/CompilationNotInc#g' %S/Inputs/compile_commands.json > %T/CompilationNotInc/compile_commands.json

// Check that no files are transformed when -p is specified but not -include.
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationNotInc/a1
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationNotInc/a2
// RUN: cp %S/Inputs/compilations.cpp %T/CompilationNotInc/a3

// RUN: not clang-modernize -use-nullptr -p=%T/CompilationNotInc
// RUN: not diff -b %T/compilations_expected.cpp %T/CompilationNotInc/a1/compilations.cpp
// RUN: not diff -b %T/compilations_expected.cpp %T/CompilationNotInc/a2/compilations.cpp
// RUN: not diff -b %T/compilations_expected.cpp %T/CompilationNotInc/a3/compilations.cpp
