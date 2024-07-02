// Verify Driver passes on -fsanitize-metadata-ignorelist.

// RUN: echo "fun:foo" > %t.1
// RUN: echo "fun:bar" > %t.2

// RUN: %clang --target=x86_64-linux-gnu -fexperimental-sanitize-metadata=all -fexperimental-sanitize-metadata-ignorelist=%t.1 -fexperimental-sanitize-metadata-ignorelist=%t.2 %s -### 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-linux-gnu -fexperimental-sanitize-metadata=atomics -fexperimental-sanitize-metadata-ignorelist=%t.1 -fexperimental-sanitize-metadata-ignorelist=%t.2 %s -### 2>&1 | FileCheck %s
// CHECK: "-fexperimental-sanitize-metadata-ignorelist={{.*}}.1" "-fexperimental-sanitize-metadata-ignorelist={{.*}}.2"

// Verify -fsanitize-metadata-ignorelist flag not passed if there is no -fsanitize-metadata flag.
// RUN: %clang --target=x86_64-linux-gnu -fexperimental-sanitize-metadata-ignorelist=%t.1 -fexperimental-sanitize-metadata-ignorelist=%t.2 %s -### 2>&1 | FileCheck %s --check-prefix=NOSANMD
// RUN: %clang --target=aarch64-linux-gnu -fexperimental-sanitize-metadata-ignorelist=%t.1 -fexperimental-sanitize-metadata-ignorelist=%t.2 %s -### 2>&1 | FileCheck %s --check-prefix=NOSANMD
// NOSANMD: warning: argument unused during compilation: '-fexperimental-sanitize-metadata-ignorelist
// NOSANMD-NOT: "-fexperimental-sanitize-metadata-ignorelist
