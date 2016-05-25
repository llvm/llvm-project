// Sanitizers need to unwind stack at any code location.
// Test that unwind tables are enabled in supported configurations.

// RUN: %clang -target x86_64-linux-gnu -fsanitize=address %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target i686-linux-gnu -fsanitize=address %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target arm-linux-androideabi -fsanitize=address %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -fsanitize=dataflow %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-cache-frag %s -### 2>&1 |  FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -fsanitize=efficiency-working-set %s -### 2>&1 |  FileCheck %s

// CHECK: -munwind-tables
