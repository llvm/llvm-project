// Checks that -fmodules-driver correctly handles compilations using
// Standard C++20 modules.

// RUN: split-file %s %t

// RUN: %clang -c -std=c++23 \
// RUN:   -fmodules-driver -Rmodules-driver -Rmodule-import \
// RUN:   %t/main.cpp %t/A.cppm %t/A-part1.cppm %t/A-part1-impl.cppm %t/B.cppm 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefix=CHECK-REMARKS %s

// CHECK-REMARKS: [[PREFIX]]/A.cppm:2:8: remark: importing module 'A:part1' from
// CHECK-REMARKS: [[PREFIX]]/A.cppm:3:1: remark: importing module 'B' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'A' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'A:part1' into 'A' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'B' into 'A' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:2:1: remark: importing module 'B' from

// RUN: %clang -std=c++23 \
// RUN:   -fmodules-driver -Rmodules-driver -Rmodule-import \
// RUN:   %t/main.cpp %t/A.cppm %t/A-part1.cppm %t/A-part1-impl.cppm %t/B.cppm \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck --check-prefix=CHECK-CC1 %s

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "{{.*}}/B.cppm"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[B_PCM:[^"]+]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "{{.*}}/A-part1-impl.cppm"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[A_PART1_IMPL_PCM:[^"]+]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "{{.*}}/A-part1.cppm"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[A_PART1_PCM:[^"]+]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "{{.*}}/A.cppm"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[A_PCM:[^"]+]]"
// CHECK-CC1-SAME: "-fmodule-file=A:part1=[[A_PART1_PCM]]"
// CHECK-CC1-SAME: "-fmodule-file=B=[[B_PCM]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "{{.*}}/main.cpp"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-file=A=[[A_PCM]]"
// CHECK-CC1-SAME: "-fmodule-file=A:part1=[[A_PART1_PCM]]"
// CHECK-CC1-SAME: "-fmodule-file=B=[[B_PCM]]"

//--- main.cpp
import A;
import B;

int main() {
  return a() + b();
}

//--- A.cppm
export module A;
export import :part1;
import B;

export int a() {
  return part1() + b();
}

//--- A-part1.cppm
export module A:part1;
export int part1();

//--- A-part1-impl.cppm
module A:part1_impl;

int part1() {
  return 30;
}

//--- B.cppm
export module B;

export int b() {
  return 12;
}
