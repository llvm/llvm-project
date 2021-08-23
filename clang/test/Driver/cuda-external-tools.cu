// Tests that ptxas and fatbinary are invoked correctly during CUDA
// compilation.
//
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Regular compiles with -O{0,1,2,3,4,fast}.  -O4 and -Ofast map to ptxas O3.
// RUN: %clang -### -target x86_64-linux-gnu -O0 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT0 %s
// RUN: %clang -### -target x86_64-linux-gnu -O1 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT1 %s
// RUN: %clang -### -target x86_64-linux-gnu -O2 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT2 %s
// RUN: %clang -### -target x86_64-linux-gnu -O3 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT3 %s
// RUN: %clang -### -target x86_64-linux-gnu -O4 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT3 %s
// RUN: %clang -### -target x86_64-linux-gnu -Ofast -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT3 %s
// Generating relocatable device code
// RUN: %clang -### -target x86_64-linux-gnu -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,RDC %s

// With debugging enabled, ptxas should be run with with no ptxas optimizations.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-noopt-device-debug -O2 -g -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,DBG %s

// --no-cuda-noopt-device-debug overrides --cuda-noopt-device-debug.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-noopt-device-debug \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:   --no-cuda-noopt-device-debug -O2 -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT2 %s

// Regular compile without -O.  This should result in us passing -O0 to ptxas.
// RUN: %clang -### -target x86_64-linux-gnu -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT0 %s

// Regular compiles with -Os and -Oz.  For lack of a better option, we map
// these to ptxas -O3.
// RUN: %clang -### -target x86_64-linux-gnu -Os -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT2 %s
// RUN: %clang -### -target x86_64-linux-gnu -Oz -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT2 %s

// Regular compile targeting sm_35.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35 %s
// Separate compilation targeting sm_35.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,RDC %s

// 32-bit compile.
// RUN: %clang -### -target i386-linux-gnu -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH32,SM35 %s
// 32-bit compile when generating relocatable device code.
// RUN: %clang -### -target i386-linux-gnu -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH32,SM35,RDC %s

// Compile with -fintegrated-as.  This should still cause us to invoke ptxas.
// RUN: %clang -### -target x86_64-linux-gnu -fintegrated-as -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT0 %s
// Check that we still pass -c when generating relocatable device code.
// RUN: %clang -### -target x86_64-linux-gnu -fintegrated-as -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,RDC %s

// Check -Xcuda-ptxas and -Xcuda-fatbinary
// RUN: %clang -### -target x86_64-linux-gnu -c -Xcuda-ptxas -foo1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:   -Xcuda-fatbinary -bar1 -Xcuda-ptxas -foo2 -Xcuda-fatbinary -bar2 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK,SM35,PTXAS-EXTRA,FATBINARY-EXTRA %s

// MacOS spot-checks
// RUN: %clang -### -target x86_64-apple-macosx -O0 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,OPT0 %s
// RUN: %clang -### -target x86_64-apple-macosx --cuda-gpu-arch=sm_35 -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35 %s
// RUN: %clang -### -target i386-apple-macosx -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH32,SM35 %s

// Check relocatable device code generation on MacOS.
// RUN: %clang -### -target x86_64-apple-macosx -O0 -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,RDC %s
// RUN: %clang -### -target x86_64-apple-macosx --cuda-gpu-arch=sm_35 -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH64,SM35,RDC %s
// RUN: %clang -### -target i386-apple-macosx -fgpu-rdc -c %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefixes=CHECK,ARCH32,SM35,RDC %s

// Check that CLANG forwards the -v flag to PTXAS.
// RUN: %clang -### -save-temps -no-canonical-prefixes -v %s 2>&1 \
// RUN:   --offload-arch=sm_35 --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: | FileCheck -check-prefix=CHK-PTXAS-VERBOSE %s

// Match clang job that produces PTX assembly.
// CHECK: "-cc1"
// ARCH64-SAME: "-triple" "nvptx64-nvidia-cuda"
// ARCH32-SAME: "-triple" "nvptx-nvidia-cuda"
// SM35-SAME: "-target-cpu" "sm_35"
// RDC-SAME: "-fgpu-rdc"
// CHECK-NOT: "-fgpu-rdc"
// SM35-SAME: "-o" "[[PTXFILE:[^"]*]]"

// Match the call to ptxas (which assembles PTX to SASS).
// CHECK: ptxas
// ARCH64-SAME: "-m64"
// ARCH32-SAME: "-m32"
// OPT0-SAME: "-O0"
// OPT0-NOT: "-g"
// OPT1-SAME: "-O1"
// OPT1-NOT: "-g"
// OPT2-SAME: "-O2"
// OPT2-NOT: "-g"
// OPT3-SAME: "-O3"
// OPT3-NOT: "-g"
// DBG-SAME: "-g" "--dont-merge-basicblocks" "--return-at-end"
// SM35-SAME: "--gpu-name" "sm_35"
// SM35-SAME: "--output-file" "[[CUBINFILE:[^"]*]]"
// CHECK-SAME: "[[PTXFILE]]"
// PTXAS-EXTRA-SAME: "-foo1"
// PTXAS-EXTRA-SAME: "-foo2"
// RDC-SAME: "-c"
// CHECK-NOT: "-c"

// Match the call to fatbinary (which combines all our PTX and SASS into one
// blob).
// CHECK: fatbinary
// CHECK-SAME-DAG: "--cuda"
// ARCH64-SAME-DAG: "-64"
// ARCH32-SAME-DAG: "-32"
// CHECK-DAG: "--create" "[[FATBINARY:[^"]*]]"
// SM35-SAME-DAG: "--image=profile=compute_35,file=[[PTXFILE]]"
// SM35-SAME-DAG: "--image=profile=sm_35,file=[[CUBINFILE]]"
// FATBINARY-EXTRA-SAME: "-bar1"
// FATBINARY-EXTRA-SAME: "-bar2"

// Match the clang job for host compilation.
// CHECK: "-cc1"
// ARCH64-SAME: "-triple" "x86_64-
// ARCH32-SAME: "-triple" "i386-
// CHECK-SAME: "-fcuda-include-gpubinary" "[[FATBINARY]]"
// RDC-SAME: "-fgpu-rdc"
// CHECK-NOT: "-fgpu-rdc"

// CHK-PTXAS-VERBOSE: ptxas{{.*}}" "-v"
