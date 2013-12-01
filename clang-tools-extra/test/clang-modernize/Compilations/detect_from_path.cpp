// The following block tests:
//   - A compilation database is detected from build path specified by -p and
//     files are provided.

// Create directory structure
// a1, a2  and a3 are specified paths for files in the compilation database.
// RUN: rm -rf %T/DetectFromPath
// RUN: mkdir -p %T/DetectFromPath
// RUN: mkdir -p %T/DetectFromPath/a1
// RUN: mkdir -p %T/DetectFromPath/a2
// RUN: mkdir -p %T/DetectFromPath/a3

// This test uses a compilation database
// RUN: sed -e 's#$(path)#%/T/DetectFromPath#g' %S/Inputs/compile_commands.json > %T/DetectFromPath/compile_commands.json

// Check that files are transformed when -p is provided and files are specified.
// RUN: cp %S/Inputs/compilations.cpp %T/DetectFromPath/a1
// RUN: cp %S/Inputs/compilations.cpp %T/DetectFromPath/a2
// RUN: cp %S/Inputs/compilations.cpp %T/DetectFromPath/a3

// RUN: clang-modernize -use-nullptr -p=%T/DetectFromPath %T/DetectFromPath/a1/compilations.cpp %T/DetectFromPath/a3/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/DetectFromPath/a1/compilations.cpp
// RUN: not diff -b %S/Inputs/compilations_expected.cpp %T/DetectFromPath/a2/compilations.cpp
// RUN: diff -b %S/Inputs/compilations_expected.cpp %T/DetectFromPath/a3/compilations.cpp
