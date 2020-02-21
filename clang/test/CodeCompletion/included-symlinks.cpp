// REQUIRES: shell
// RUN: rm -rf %t && mkdir -p %t/real/myproj && mkdir -p %t/links
// RUN: touch %t/real/foo.h && ln -s %t/real/foo.h %t/links/foo.h
// RUN: touch %t/real/foobar.h && ln -s %t/real/foobar.h %t/links/foobar.h
// RUN: touch %t/real/myproj/test.h && ln -s %t/real/myproj %t/links/myproj

// Suggest symlinked header files.
#include "foo.h"
// RUN: %clang -fsyntax-only -I%t/links -Xclang -code-completion-at=%s:8:13 %s | FileCheck -check-prefix=CHECK-1 %s
// CHECK-1: foo.h"
// CHECK-1: foobar.h"

// Suggest symlinked folder.
#include "mypr"
// RUN: %clang -fsyntax-only -I%t/links -Xclang -code-completion-at=%s:14:13 %s | FileCheck -check-prefix=CHECK-2 %s
// CHECK-2: myproj/
