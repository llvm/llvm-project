// UNSUPPORTED: system-windows
// Test uses Unix absolute paths.
// Verify that relative paths are made absolute and dots are removed.
// RUN: %clang -### -fsanitize=address -ffile-compilation-dir=. %s 2>&1 | FileCheck %s --check-prefix=RELATIVE
// RUN: %clang -### -fsanitize=address -ffile-compilation-dir=/foo/bar/.. %s 2>&1 | FileCheck %s --check-prefix=DOTS

// RELATIVE-NOT: "-fsanitize-compilation-dir=."
// RELATIVE: "-fsanitize-compilation-dir={{/[^"]+}}"
// DOTS: "-fsanitize-compilation-dir=/foo"
