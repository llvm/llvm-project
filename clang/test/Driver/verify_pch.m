// RUN: touch %t.pch
// RUN: %clang -### -verify-pch %t.pch 2> %t.log.1
// RUN: FileCheck %s < %t.log.1
// CHECK: -verify-pch

// Also ensure that the language setting is not affected by the .pch extension
// CHECK-NOT: "-x" "precompiled-header"

// RUN: %clang -### -verify-pch -x objective-c %t.pch 2> %t.log.2
// RUN: FileCheck -check-prefix=CHECK2 %s < %t.log.2
// CHECK2: "-x" "objective-c"
// CHECK2-NOT: "-x" "precompiled-header"
