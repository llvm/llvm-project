// RUN: rm -rf %t && mkdir -p %t/Foo.framework/Headers/SubFolder && mkdir %t/NotAFramework/
// RUN: touch %t/Foo.framework/Headers/Foo.h && touch %t/Foo.framework/Headers/FOOClass.h
// RUN: touch %t/Foo.framework/Headers/SubFolder/FOOInternal.h

#import <Foo/Foo.h>

#import <Foo/SubFolder/FOOInternal.h>

// Note: the run lines follow their respective tests, since line/column
// matter in this test.

// Autocomplete frameworks without the ".framework" extension.
//
// RUN: %clang -fsyntax-only -F %t -Xclang -code-completion-at=%s:5:10 %s -o - | FileCheck -check-prefix=CHECK-1 %s
// CHECK-1-NOT: Foo.framework/
// CHECK-1-NOT: NotAFramework/
// CHECK-1: Foo/

// Autocomplete for frameworks inside its Headers folder.
//
// RUN: %clang -fsyntax-only -F %t -Xclang -code-completion-at=%s:5:14 %s -o - | FileCheck -check-prefix=CHECK-2 %s
// CHECK-2: Foo.h>
// CHECK-2: FOOClass.h>
// CHECK-2: SubFolder/

// Autocomplete for folders inside of a frameworks.
//
// RUN: %clang -fsyntax-only -F %t -Xclang -code-completion-at=%s:7:24 %s -o - | FileCheck -check-prefix=CHECK-3 %s
// CHECK-3: FOOInternal.h>
