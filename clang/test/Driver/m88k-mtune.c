// RUN: %clang -target m88k-openbsd -mtune=generic -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-GENERIC %s
// CHECK-GENERIC: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88000"

// RUN: %clang -target m88k-openbsd -mtune=mc88000 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MC88000 %s
// CHECK-MC88000: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88000"

// RUN: %clang -target m88k-openbsd -mtune=m88000 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-M88000 %s
// CHECK-M88000: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88000"

// RUN: %clang -target m88k-openbsd -mtune=mc88100 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MC88100 %s
// CHECK-MC88100: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88100"

// RUN: %clang -target m88k-openbsd -mtune=m88100 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-M88100 %s
// CHECK-M88100: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88100"

// RUN: %clang -target m88k-openbsd -mtune=mc88110 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MC88110 %s
// CHECK-MC88110: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88110"

// RUN: %clang -target m88k-openbsd -mtune=m88110 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-M88110 %s
// CHECK-M88110: "-cc1"{{.*}} "-triple" "m88k-{{.*}} "-tune-cpu" "mc88110"

