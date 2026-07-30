// RUN: %clang %s -### -o %t.o -fsanitize-undefined-strip-path-components=42 2>&1 | FileCheck %s
// CHECK: "-fsanitize-undefined-strip-path-components=42"
