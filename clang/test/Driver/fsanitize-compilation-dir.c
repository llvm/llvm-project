// RUN: %clang -### -fsanitize=address -fsanitize-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=SANITIZE
// RUN: %clang -### -fsanitize=undefined -fsanitize-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=SANITIZE
// RUN: %clang -### -fsanitize=address -ffile-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=FILE
// RUN: %clang -### -fsanitize=undefined -ffile-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=FILE

// Verify that -fsanitize-compilation-dir wins over -ffile-compilation-dir when it comes last.
// RUN: %clang -### -fsanitize=address -ffile-compilation-dir=/bar -fsanitize-compilation-dir=/foo %s 2>&1 | FileCheck %s --check-prefix=OVERRIDE
// RUN: %clang -### -fsanitize=address -fsanitize-compilation-dir=/foo -ffile-compilation-dir=/bar %s 2>&1 | FileCheck %s --check-prefix=OVERRIDE-FILE

// SANITIZE: "-fsanitize-compilation-dir=/foo"
// FILE: "-fsanitize-compilation-dir=/foo"
// OVERRIDE: "-fsanitize-compilation-dir=/foo"
// OVERRIDE-FILE: "-fsanitize-compilation-dir=/bar"
