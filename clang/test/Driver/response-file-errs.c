// If response file does not exist, '@file; directive remains unexpanded in
// command line.
//
// RUN: not %clang @%S/Inputs/inexistent.rsp -### 2>&1 | FileCheck --check-prefix=INEXISTENT %s
// INEXISTENT: @{{.*}}Inputs/inexistent.rsp

// As the above case but '@file' is in response file.
//
// RUN: not %clang @%S/Inputs/inc-inexistent.rsp -### 2>&1 | FileCheck --check-prefix=INEXISTENT2 %s
// INEXISTENT2: @{{.*}}inexistent.txt

// If file in `@file` is a directory, it is an error.
//
// RUN: not %clang @%S/Inputs -### 2>&1 | FileCheck --check-prefix=DIRECTORY %s
// DIRECTORY: cannot not open file '{{.*}}Inputs': {{[Ii]}}s a directory
