// Ensure we support the -mtune flag.
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -mtune=nocona 2>&1 \
// RUN:   | FileCheck %s -check-prefix=nocona
// nocona: "-tune-cpu" "nocona"

// Unlike march we allow 32-bit only cpus with mtune.

// RUN: %clang -target x86_64-unknown-unknown -c -### %s -mtune=i686 2>&1 \
// RUN:   | FileCheck %s -check-prefix=i686
// i686: "-tune-cpu" "i686"

// RUN: %clang -target x86_64-unknown-unknown -c -### %s -mtune=pentium4 2>&1 \
// RUN:   | FileCheck %s -check-prefix=pentium4
// pentium4: "-tune-cpu" "pentium4"

// RUN: %clang -target x86_64-unknown-unknown -c -### %s -mtune=athlon 2>&1 \
// RUN:   | FileCheck %s -check-prefix=athlon
// athlon: "-tune-cpu" "athlon"

