// Check that -save-temps doesn't name an intermediate file after the input
// file, which would overwrite it. On Darwin, a .s file is preprocessed, and
// the preprocessed output's name ends in .s.

// RUN: rm -rf %t && mkdir -p %t/sub %t/obj && cd %t
// RUN: touch t.s sub/t.s

// The output would be the input in the current directory.
// RUN: %clang --target=x86_64-apple-darwin -c -save-temps t.s -### 2>&1 | FileCheck %s --check-prefix=CWD
// CWD: "-E"
// CWD-NOT: "-o" "t.s"
// CWD-SAME: "-o" "{{[^"]*}}t-{{[^"]*}}.s"

// With -save-temps=obj, the output would be the input in the output directory.
// RUN: %clang --target=x86_64-apple-darwin -c -save-temps=obj sub/t.s -o sub/t.o -### 2>&1 | FileCheck %s --check-prefix=OBJ
// OBJ: "-E"
// OBJ-NOT: "-o" "sub{{/|\\\\}}t.s"
// OBJ-SAME: "-o" "{{[^"]*}}t-{{[^"]*}}.s"

// An output with the input's name in another directory is a different file.
// RUN: cd obj && %clang --target=x86_64-apple-darwin -c -save-temps ../t.s -o t.o -### 2>&1 | FileCheck %s --check-prefix=OTHER-DIR
// OTHER-DIR: "-E"
// OTHER-DIR-SAME: "-o" "t.s"
