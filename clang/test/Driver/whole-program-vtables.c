// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -O1 -### %s 2>&1 | FileCheck --check-prefix=WPD-NO-LTO %s
// RUN: %clang_cl --target=x86_64-pc-win32 -fwhole-program-vtables -O1 -### -- %s 2>&1 | FileCheck --check-prefix=WPD-NO-LTO %s
// WPD-NO-LTO: "-fwhole-program-vtables"

// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -flto -### %s 2>&1 | FileCheck --check-prefix=LTO %s
// RUN: not %clang_cl --target=x86_64-pc-win32 -fwhole-program-vtables -flto -### -- %s 2>&1 | FileCheck --check-prefix=LTO %s
// LTO: "-fwhole-program-vtables"

// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -fno-whole-program-vtables -flto -### %s 2>&1 | FileCheck --check-prefix=LTO-DISABLE %s
// RUN: not %clang_cl --target=x86_64-pc-win32 -fwhole-program-vtables -fno-whole-program-vtables -flto -### -- %s 2>&1 | FileCheck --check-prefix=LTO-DISABLE %s
// LTO-DISABLE-NOT: "-fwhole-program-vtables"
