// Verify that -ffile-compilation-dir= emits -fsanitize-compilation-dir= to cc1.
// RUN: %clang -### -fsanitize=address -ffile-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=FILE
// RUN: %clang -### -fsanitize=undefined -ffile-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=FILE

// FILE: "-fsanitize-compilation-dir=/foo"
