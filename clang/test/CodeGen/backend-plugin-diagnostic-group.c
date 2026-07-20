// Tests that a backend (pass) plugin diagnostic which names its own warning
// group via llvm::DiagnosticInfo::getWarningGroup() is controlled by the user
// exactly like a frontend plugin's diagnostic: it prints "[-W<group>]", is
// silenced by -Wno-<group>, the -Wno-plugin umbrella and the
// -Wno-user-defined-warnings root, and is promoted by -Werror=<group>. The Bye
// example pass emits such a diagnostic under -bye-warn.

// The warning is on by default and prints its group.
// RUN: %clang_cc1 -emit-llvm -o /dev/null -O2 \
// RUN:   -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -bye-warn %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN %s

// It is silenced by -Wno-<group>, by the -Wno-plugin umbrella, and by the
// -Wno-user-defined-warnings root over every runtime group.
// RUN: %clang_cc1 -emit-llvm -o /dev/null -O2 \
// RUN:   -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -bye-warn \
// RUN:   -Wno-bye-plugin %s 2>&1 | FileCheck --allow-empty --check-prefix=SILENT %s
// RUN: %clang_cc1 -emit-llvm -o /dev/null -O2 \
// RUN:   -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -bye-warn \
// RUN:   -Wno-plugin %s 2>&1 | FileCheck --allow-empty --check-prefix=SILENT %s
// RUN: %clang_cc1 -emit-llvm -o /dev/null -O2 \
// RUN:   -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -bye-warn \
// RUN:   -Wno-user-defined-warnings %s 2>&1 | FileCheck --allow-empty --check-prefix=SILENT %s

// It is promoted by -Werror=<group>.
// RUN: not %clang_cc1 -emit-llvm -o /dev/null -O2 \
// RUN:   -fpass-plugin=%llvmshlibdir/Bye%pluginext -mllvm -bye-warn \
// RUN:   -Werror=bye-plugin %s 2>&1 | FileCheck --check-prefix=WERROR %s

// REQUIRES: plugins, llvm-examples
// UNSUPPORTED: target={{.*windows.*}}
// Plugins are currently broken on AIX, at least in the CI.
// XFAIL: target={{.*}}-aix{{.*}}

void f(void) {}

// WARN: warning: Bye saw function 'f' [-Wbye-plugin]
// SILENT-NOT: Bye saw function
// WERROR: error: Bye saw function 'f'
