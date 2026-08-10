// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av8 2>&1 | \
// RUN:   FileCheck -check-prefix=V8 %s
// V8: -cc1as
// V8: "-target-feature" "-v8plus"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av8plus 2>&1 | \
// RUN:   FileCheck -check-prefix=V8PLUS %s
// V8PLUS: -cc1as
// V8PLUS: "-target-feature" "+v8plus"
// V8PLUS: "-target-feature" "+v9"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av8plusa 2>&1 | \
// RUN:   FileCheck -check-prefix=V8PLUSA %s
// V8PLUSA: -cc1as
// V8PLUSA: "-target-feature" "+v8plus"
// V8PLUSA: "-target-feature" "+v9"
// V8PLUSA: "-target-feature" "+vis"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av8plusb 2>&1 | \
// RUN:   FileCheck -check-prefix=V8PLUSB %s
// V8PLUSB: -cc1as
// V8PLUSB: "-target-feature" "+v8plus"
// V8PLUSB: "-target-feature" "+v9"
// V8PLUSB: "-target-feature" "+vis"
// V8PLUSB: "-target-feature" "+vis2"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av8plusd 2>&1 | \
// RUN:   FileCheck -check-prefix=V8PLUSD %s
// V8PLUSD: -cc1as
// V8PLUSD: "-target-feature" "+v8plus"
// V8PLUSD: "-target-feature" "+v9"
// V8PLUSD: "-target-feature" "+vis"
// V8PLUSD: "-target-feature" "+vis2"
// V8PLUSD: "-target-feature" "+vis3"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av9 2>&1 | \
// RUN:   FileCheck -check-prefix=V9 %s
// V9: -cc1as
// V9: "-target-feature" "+v9"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av9a 2>&1 | \
// RUN:   FileCheck -check-prefix=V9A %s
// V9A: -cc1as
// V9A: "-target-feature" "+v9"
// V9A: "-target-feature" "+vis"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av9b 2>&1 | \
// RUN:   FileCheck -check-prefix=V9B %s
// V9B: -cc1as
// V9B: "-target-feature" "+v9"
// V9B: "-target-feature" "+vis"
// V9B: "-target-feature" "+vis2"

// RUN: %clang --target=sparc-linux-gnu -### -fintegrated-as -c %s -Wa,-Av9d 2>&1 | \
// RUN:   FileCheck -check-prefix=V9D %s
// V9D: -cc1as
// V9D: "-target-feature" "+v9"
// V9D: "-target-feature" "+vis"
// V9D: "-target-feature" "+vis2"
// V9D: "-target-feature" "+vis3"

// RUN: %clang --target=sparc64-linux-gnu -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=VIS-DEFAULT %s
// RUN: %clang --target=sparc64-freebsd -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=VIS-DEFAULT %s
// RUN: %clang --target=sparc64-openbsd -### -fintegrated-as -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=VIS-DEFAULT %s
// VIS-DEFAULT: -cc1as
// VIS-DEFAULT: "-target-feature" "+vis"
