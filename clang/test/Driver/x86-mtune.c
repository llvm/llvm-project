// Ensure we support the -mtune flag.

// cpu: "-target-cpu" "x86-64"

// Default mtune should be generic.
// RUN: %clang -### -c --target=x86_64 %s 2>&1 | FileCheck %s -check-prefix=generic

// RUN: %clang -### -c --target=x86_64 %s -mtune=generic 2>&1 | FileCheck %s --check-prefixes=cpu,generic
// generic: "-tune-cpu" "generic"

// RUN: %clang -### -c --target=x86_64 %s -mtune=nocona 2>&1 | FileCheck %s --check-prefixes=cpu,nocona
// nocona: "-tune-cpu" "nocona"

// Unlike march we allow 32-bit only cpus with mtune.

// RUN: %clang -### -c --target=x86_64 %s -mtune=i686 2>&1 | FileCheck %s --check-prefixes=cpu,i686
// i686: "-tune-cpu" "i686"

// RUN: %clang -### -c --target=x86_64 %s -mtune=pentium4 2>&1 | FileCheck %s -check-prefix=pentium4
// pentium4: "-tune-cpu" "pentium4"

// RUN: %clang -### -c --target=x86_64 %s -mtune=athlon 2>&1 | FileCheck %s -check-prefixes=cpu,athlon
// athlon: "-tune-cpu" "athlon"

// Check interaction between march and mtune.

// -march should remove default mtune generic.
// RUN: %clang -### -c --target=x86_64 %s -march=core2 2>&1 | FileCheck %s -check-prefix=marchcore2
// marchcore2: "-target-cpu" "core2"
// marchcore2-NOT: "-tune-cpu"

// -march should remove default mtune generic.
// RUN: %clang -### -c --target=x86_64 %s -march=core2 -mtune=nehalem 2>&1 | FileCheck %s -check-prefix=marchmtune
// marchmtune: "-target-cpu" "core2"
// mmarchmtune: "-tune-cpu" "nehalem"

// RUN: not %clang %s -target x86_64 -E -mtune=x86-64-v2 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang %s -target x86_64 -E -mtune=x86-64-v3 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang %s -target x86_64 -E -mtune=x86-64-v4 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: error: unknown target CPU '{{.*}}'
