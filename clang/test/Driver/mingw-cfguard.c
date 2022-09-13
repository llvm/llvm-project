// RUN: %clang --target=x86_64-w64-windows-gnu -### %s 2>&1 | FileCheck -check-prefixes=NO_CF,DEFAULT %s
// RUN: %clang --target=x86_64-w64-windows-gnu -### %s -mguard=none 2>&1 | FileCheck -check-prefixes=NO_CF,GUARD_NONE %s
// NO_CF: "-cc1"
// NO_CF-NOT: "-cfguard"
// NO_CF-NOT: "-cfguard-no-checks"
// NO_CF-NEXT: ld{{(.lld)?}}{{(.exe)?}}"
// NO_CF-NOT: "--guard-cf"
// DEFAULT-NOT: "--no-guard-cf"
// GUARD_NONE-SAME: "--no-guard-cf"

// RUN: %clang --target=x86_64-w64-windows-gnu -### %s -mguard=cf 2>&1 | FileCheck -check-prefix=GUARD_CF %s
// GUARD_CF: "-cc1"
// GUARD_CF-SAME: "-cfguard"
// GUARD_CF-NEXT: ld{{(.lld)?}}{{(.exe)?}}"
// GUARD_CF-SAME: "--guard-cf"
// GUARD_CF-NOT: "--no-guard-cf"

// RUN: %clang --target=x86_64-w64-windows-gnu -### %s -mguard=cf-nochecks 2>&1 | FileCheck -check-prefix=GUARD_NOCHECKS %s
// GUARD_NOCHECKS: "-cc1"
// GUARD_NOCHECKS-NOT: "-cfguard"
// GUARD_NOCHECKS-SAME: "-cfguard-no-checks"
// GUARD_NOCHECKS-NOT: "-cfguard"
// GUARD_NOCHECKS-NEXT: ld{{(.lld)?}}{{(.exe)?}}"
// GUARD_NOCHECKS-SAME: "--guard-cf"
// GUARD_NOCHECKS-NOT: "--no-guard-cf"

// RUN: %clang --target=x86_64-w64-windows-gnu -### %s -mguard=xxx 2>&1 | FileCheck -check-prefix=GUARD_UNKNOWN %s
// GUARD_UNKNOWN: error: unsupported argument 'xxx' to option '--mguard='
