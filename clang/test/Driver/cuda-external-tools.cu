// Tests that ptxas and fatbinary are correctly during CUDA compilation.
//
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Regular compiles with -O{0,1,2,3,4,fast}.  -O4 and -Ofast map to ptxas O3.
// RUN: %clang -### -target x86_64-linux-gnu -O0 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT0 %s
// RUN: %clang -### -target x86_64-linux-gnu -O1 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT1 %s
// RUN: %clang -### -target x86_64-linux-gnu -O2 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT2 %s
// RUN: %clang -### -target x86_64-linux-gnu -O3 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT3 %s
// RUN: %clang -### -target x86_64-linux-gnu -O4 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT3 %s
// RUN: %clang -### -target x86_64-linux-gnu -Ofast -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT3 %s

// With debugging enabled, ptxas should be run with with no ptxas optimizations.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-noopt-device-debug -O2 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix DBG %s

// --no-cuda-noopt-device-debug overrides --cuda-noopt-device-debug.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-noopt-debug \
// RUN:   --no-cuda-noopt-debug -O2 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT2 %s

// Regular compile without -O.  This should result in us passing -O0 to ptxas.
// RUN: %clang -### -target x86_64-linux-gnu -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT0 %s

// Regular compiles with -Os and -Oz.  For lack of a better option, we map
// these to ptxas -O3.
// RUN: %clang -### -target x86_64-linux-gnu -Os -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT2 %s
// RUN: %clang -### -target x86_64-linux-gnu -Oz -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT2 %s

// Regular compile targeting sm_35.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM35 %s

// 32-bit compile.
// RUN: %clang -### -target x86_32-linux-gnu -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH32 -check-prefix SM20 %s

// Compile with -fintegrated-as.  This should still cause us to invoke ptxas.
// RUN: %clang -### -target x86_64-linux-gnu -fintegrated-as -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCH64 -check-prefix SM20 -check-prefix OPT0 %s

// Check -Xcuda-ptxas and -Xcuda-fatbinary
// RUN: %clang -### -target x86_64-linux-gnu -c -Xcuda-ptxas -foo1 \
// RUN:   -Xcuda-fatbinary -bar1 -Xcuda-ptxas -foo2 -Xcuda-fatbinary -bar2 %s 2>&1 \
// RUN: | FileCheck -check-prefix SM20 -check-prefix PTXAS-EXTRA \
// RUN:   -check-prefix FATBINARY-EXTRA %s

// Match clang job that produces PTX assembly.
// CHECK: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// SM20: "-target-cpu" "sm_20"
// SM35: "-target-cpu" "sm_35"
// SM20: "-o" "[[PTXFILE:[^"]*]]"
// SM35: "-o" "[[PTXFILE:[^"]*]]"

// Match the call to ptxas (which assembles PTX to SASS).
// CHECK:ptxas
// ARCH64: "-m64"
// ARCH32: "-m32"
// OPT0: "-O0"
// OPT0-NOT: "-g"
// OPT1: "-O1"
// OPT1-NOT: "-g"
// OPT2: "-O2"
// OPT2-NOT: "-g"
// OPT3: "-O3"
// OPT3-NOT: "-g"
// DBG: "-g" "--dont-merge-basicblocks" "--return-at-end"
// SM20: "--gpu-name" "sm_20"
// SM35: "--gpu-name" "sm_35"
// SM20: "--output-file" "[[CUBINFILE:[^"]*]]"
// SM35: "--output-file" "[[CUBINFILE:[^"]*]]"
// PTXAS-EXTRA: "-foo1"
// PTXAS-EXTRA-SAME: "-foo2"
// CHECK-SAME: "[[PTXFILE]]"

// Match the call to fatbinary (which combines all our PTX and SASS into one
// blob).
// CHECK:fatbinary
// CHECK-DAG: "--cuda"
// ARCH64-DAG: "-64"
// ARCH32-DAG: "-32"
// CHECK-DAG: "--create" "[[FATBINARY:[^"]*]]"
// SM20-DAG: "--image=profile=compute_20,file=[[PTXFILE]]"
// SM35-DAG: "--image=profile=compute_35,file=[[PTXFILE]]"
// SM20-DAG: "--image=profile=sm_20,file=[[CUBINFILE]]"
// SM35-DAG: "--image=profile=sm_35,file=[[CUBINFILE]]"
// FATBINARY-EXTRA: "-bar1"
// FATBINARY-EXTRA-SAME: "-bar2"

// Match the clang job for host compilation.
// CHECK: "-cc1" "-triple" "x86_64--linux-gnu"
// CHECK-SAME: "-fcuda-include-gpubinary" "[[FATBINARY]]"
