// Check driver works
// RUN: %clang -x objective-c-header -fsyntax-only -fpch-validate-input-files-content %s -### 2>&1 | FileCheck --check-prefix=CHECK-CC1 %s
// CHECK-CC1: -fvalidate-ast-input-files-content
//
// PCH only: Test that a mtime mismatch without content change is fine
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '// m.h' > %t/m.h
// RUN: echo '#include "m.h"' > %t/a.h
// RUN: %clang_cc1 -emit-pch -o %t/a.pch -I %t -x objective-c-header %t/a.h -fvalidate-ast-input-files-content
// RUN: touch -m -a -t 202901010000 %t/m.h
// RUN: %clang_cc1 -fsyntax-only -I %t -include-pch %t/a.pch %s -verify -fvalidate-ast-input-files-content
//
// PCH only: Test that a mtime mismatch with content change
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '// m.h' > %t/m.h
// RUN: echo '#include "m.h"' > %t/a.h
// RUN: %clang_cc1 -emit-pch -o %t/a.pch -I %t -x objective-c-header %t/a.h -fvalidate-ast-input-files-content
// RUN: echo '// m.x' > %t/m.h
// RUN: touch -m -a -t 202901010000 %t/m.h
// RUN: not %clang_cc1 -fsyntax-only -I %t -include-pch %t/a.pch %s -fvalidate-ast-input-files-content 2> %t/stderr
// RUN: FileCheck %s < %t/stderr
//
// CHECK: file '[[M_H:.*[/\\]m\.h]]' has been modified since the precompiled header '[[A_PCH:.*/a\.pch]]' was built: content changed
// CHECK: please rebuild precompiled file '[[A_PCH]]'
// expected-no-diagnostics
