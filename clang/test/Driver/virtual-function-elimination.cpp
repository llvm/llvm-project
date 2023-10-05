// RUN: not %clang -target x86_64-unknown-linux -fvirtual-function-elimination -### %s 2>&1 | FileCheck --check-prefix=BAD-LTO %s
// BAD-LTO: invalid argument '-fvirtual-function-elimination' only allowed with '-flto'

// RUN: %clang -target x86_64-unknown-linux -fvirtual-function-elimination -flto -### %s 2>&1 | FileCheck --check-prefix=GOOD %s
// RUN: %clang -target x86_64-unknown-linux -fvirtual-function-elimination -flto=full -### %s 2>&1 | FileCheck --check-prefix=GOOD %s
// RUN: %clang -target x86_64-unknown-linux -fvirtual-function-elimination -flto -fwhole-program-vtables -### %s 2>&1 | FileCheck --check-prefix=GOOD %s
// GOOD: "-fvirtual-function-elimination" "-fwhole-program-vtables"

// RUN: not %clang --target=x86_64-unknown-linux -fvirtual-function-elimination -fno-whole-program-vtables -flto -### %s 2>&1 | FileCheck --check-prefix=NO-WHOLE-PROGRAM-VTABLES %s
// NO-WHOLE-PROGRAM-VTABLES: invalid argument '-fno-whole-program-vtables' not allowed with '-fvirtual-function-elimination'
