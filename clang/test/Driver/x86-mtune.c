// Ensure we support the -mtune flag.

// Default mtune should be generic.
// RUN: %clang -target x86_64-unknown-unknown -c -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=notune
// notune: "-tune-cpu" "generic"

// RUN: %clang -target x86_64-unknown-unknown -c -### %s -mtune=generic 2>&1 \
// RUN:   | FileCheck %s -check-prefix=generic
// generic: "-tune-cpu" "generic"

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

// Check interaction between march and mtune.

// -march should remove default mtune generic.
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=core2 2>&1 \
// RUN:   | FileCheck %s -check-prefix=marchcore2
// marchcore2: "-target-cpu" "core2"
// marchcore2-NOT: "-tune-cpu"

// -march should remove default mtune generic.
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=core2 -mtune=nehalem 2>&1 \
// RUN:   | FileCheck %s -check-prefix=marchmtune
// marchmtune: "-target-cpu" "core2"
// mmarchmtune: "-tune-cpu" "nehalem"
