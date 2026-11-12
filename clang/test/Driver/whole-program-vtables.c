// RUN: not %clang -target x86_64-unknown-linux -fwhole-program-vtables -### %s 2>&1 | FileCheck --check-prefix=NO-LTO %s
// RUN: not %clang_cl --target=x86_64-pc-win32 -fwhole-program-vtables -### -- %s 2>&1 | FileCheck --check-prefix=NO-LTO %s
// NO-LTO: invalid argument '-fwhole-program-vtables' only allowed with '-flto'

// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -flto -### %s 2>&1 | FileCheck --check-prefix=LTO %s
// RUN: not %clang_cl --target=x86_64-pc-win32 -fwhole-program-vtables -flto -### -- %s 2>&1 | FileCheck --check-prefix=LTO %s
// LTO: "-fwhole-program-vtables"

/// -funified-lto does not imply -flto, so we still get an error that fwhole-program-vtables has no effect without -flto
// RUN: not %clang --target=x86_64-pc-linux-gnu -fwhole-program-vtables -funified-lto -### %s 2>&1 | FileCheck --check-prefix=NO-LTO %s
// RUN: not %clang --target=x86_64-pc-linux-gnu -fwhole-program-vtables -fno-unified-lto -### %s 2>&1 | FileCheck --check-prefix=NO-LTO %s

// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -fno-whole-program-vtables -flto -### %s 2>&1 | FileCheck --check-prefix=LTO-DISABLE %s
// RUN: not %clang_cl --target=x86_64-pc-win32 -fwhole-program-vtables -fno-whole-program-vtables -flto -### -- %s 2>&1 | FileCheck --check-prefix=LTO-DISABLE %s
// LTO-DISABLE-NOT: "-fwhole-program-vtables"
