// RUN: %clang -target arm -c -### %s -fforce-dwarf-frame 2>&1 | FileCheck --check-prefix=CHECK-ALWAYS %s
// RUN: %clang -target arm -c -### %s -fno-force-dwarf-frame 2>&1 | FileCheck --check-prefix=CHECK-NO-ALWAYS %s
// RUN: %clang -target arm -c -### %s 2>&1 | FileCheck --check-prefix=CHECK-NO-ALWAYS %s

// CHECK-ALWAYS: -fforce-dwarf-frame
// CHECK-NO-ALWAYS-NOT: -fforce-dwarf-frame
