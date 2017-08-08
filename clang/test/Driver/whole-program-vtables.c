// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -### %s 2>&1 | FileCheck --check-prefix=NO-LTO %s
// NO-LTO: invalid argument '-fwhole-program-vtables' only allowed with '-flto'
