// RUN: %clang %s -### -o %t.o -fsanitize=address -fsanitize-prefix-map=/old=/new 2>&1 | FileCheck %s --check-prefix=SANITIZE
// RUN: %clang %s -### -o %t.o -fsanitize=undefined -fsanitize-prefix-map=/old=/new 2>&1 | FileCheck %s --check-prefix=SANITIZE
// RUN: %clang %s -### -o %t.o -fsanitize=address -ffile-prefix-map=/old=/new 2>&1 | FileCheck %s --check-prefix=FILE
// RUN: %clang %s -### -o %t.o -fsanitize=undefined -ffile-prefix-map=/old=/new 2>&1 | FileCheck %s --check-prefix=FILE
// RUN: not %clang -### -fsanitize-prefix-map=old %s 2>&1 | FileCheck %s --check-prefix=INVALID
// SANITIZE: "-fsanitize-prefix-map=/old=/new"
// FILE: "-fsanitize-prefix-map=/old=/new"
// INVALID: error: invalid argument 'old' to -fsanitize-prefix-map
