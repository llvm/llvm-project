// RUN: %clang -### -c -fpass-plugin=bar.so -fplugin=bar.so -fplugin-arg-bar-option -Werror %s 2>&1 | FileCheck %s

// Plugins are only relevant for the -cc1 phase. No warning should be raised
// when applied to assembly files. See GH #88173.
// CHECK-NOT: argument unused during compilation
