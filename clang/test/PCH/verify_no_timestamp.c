// RUN: echo 'int SomeFunc() { return 42; }' > %t.h
// RUN: %clang_cc1 -Werror -fno-pch-timestamp -fvalidate-ast-input-files-content -emit-pch -o "%t.pch" %t.h

// Now change the source file, which should cause the verifier to fail with content mismatch.
// RUN: echo 'int SomeFunc() { return 13; }' > %t.h
// RUN: not %clang_cc1 -fno-pch-timestamp -fvalidate-ast-input-files-content -verify-pch %t.pch 2>&1 | FileCheck %s -DT=%t

// CHECK: fatal error: file '[[T]].h' has been modified since the precompiled header '[[T]].pch' was built: content changed
