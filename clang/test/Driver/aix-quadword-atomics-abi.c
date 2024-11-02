// RUN:  %clang -### -target powerpc-unknown-aix -S %s 2>&1 | FileCheck %s
// RUN:  %clang -### -target powerpc64-unknown-aix -S %s 2>&1 | FileCheck %s
// RUN:  %clang -### -target powerpc-unknown-aix -mabi=quadword-atomics -S \
// RUN:    %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-TARGET %s
// RUN:  %clang -### -target powerpc64-unknown-aix -mabi=quadword-atomics -S \
// RUN:    %s 2>&1 | FileCheck %s --check-prefix=CHECK-QUADWORD-ATOMICS
//
// CHECK-UNSUPPORTED-TARGET: unsupported option '-mabi=quadword-atomics' for target 'powerpc-unknown-aix'
// CHECK-NOT: "-mabi=quadword-atomics"
// CHECK-QUADWORD-ATOMICS: "-cc1"
// CHECK-QUADWORD-ATOMICS-SAME: "-mabi=quadword-atomics"
