// RUN: %clang %s --target=powerpc-unknown-aix -mno-tocdata -mtocdata -mno-tocdata -### 2>&1 | FileCheck %s -check-prefix=CHECK-FLAG1
// RUN: %clang %s --target=powerpc-unknown-aix -mno-tocdata -mtocdata -mno-tocdata -mtocdata -### 2>&1 | FileCheck %s -check-prefix=CHECK-FLAG2
// RUN: %clang %s --target=powerpc-unknown-aix -mtocdata=g1,g2 -mno-tocdata=g2 -mtocdata=g3,g4 -mno-tocdata=g5,g1 -### 2>&1 | FileCheck %s -check-prefix=CHECK-EQCONF
// RUN: %clang %s --target=powerpc-unknown-aix -mtocdata=g1 -mtocdata -mno-tocdata -mtocdata=g2,g3 -mno-tocdata=g4,g5,g3 -### 2>&1 | FileCheck %s -check-prefix=CHECK-CONF1
// RUN: %clang %s --target=powerpc-unknown-aix -mno-tocdata=g1 -mno-tocdata -mtocdata -### 2>&1 | FileCheck %s -check-prefix=CHECK-CONF2

int g1, g4, g5;
extern int g2;
int g3 = 0;
void func() {
  g2 = 0;
}

// CHECK-FLAG1-NOT: warning:
// CHECK-FLAG1: "-cc1"{{.*}}" "-mno-tocdata"

// CHECK-FLAG2-NOT: warning:
// CHECK-FLAG2: "-cc1"{{.*}}" "-mtocdata"

// CHECK-EQCONF-NOT: warning:
// CHECK-EQCONF: "-cc1"{{.*}}" "-mno-tocdata"
// CHECK-EQCONF: "-mtocdata=g3,g4"

// CHECK-CONF1-NOT: warning:
// CHECK-CONF1: "-cc1"{{.*}}" "-mno-tocdata"
// CHECK-CONF1: "-mtocdata=g2,g1"

// CHECK-CONF2-NOT: warning:
// CHECK-CONF2: "-cc1"{{.*}}" "-mtocdata"
// CHECK-CONF2: "-mno-tocdata=g1"
