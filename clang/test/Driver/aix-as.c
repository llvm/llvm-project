// General tests that as(1) invocations on AIX targets are sane. Note that we
// only test assembler functionalities in this suite.

// Check powerpc-ibm-aix7.1.0.0, system assembler is used for assembly files, 32-bit.
// RUN: %clang -x assembler %s -### -c -fintegrated-as 2>&1 \
// RUN:         --target=powerpc-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS32 %s
//
// RUN: %clang -x assembler %s -### -c 2>&1 -fno-integrated-as \
// RUN:         --target=powerpc-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS32 %s

// Check powerpc-ibm-aix7.1.0.0, 32-bit.
// RUN: %clang %s -### -c -fno-integrated-as 2>&1 \
// RUN:         --target=powerpc-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefixes=CHECK-AS32,CHECK-AS32-CC1 %s
// CHECK-AS32-NOT: warning:
// CHECK-AS32-CC1: "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-AS32: "{{.*}}as{{(.exe)?}}" 
// CHECK-AS32: "-a32" 
// CHECK-AS32: "-many" 

// Check powerpc64-ibm-aix7.1.0.0, system assembler is used for assembly files, 64-bit.
// RUN: %clang -x assembler %s -### -c -fintegrated-as 2>&1 \
// RUN:         --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS64 %s
//
// RUN: %clang -x assembler %s -### -c -fno-integrated-as 2>&1 \
// RUN:         --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS64 %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit.
// RUN: %clang %s -### -c -fno-integrated-as 2>&1 \
// RUN:         --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefixes=CHECK-AS64,CHECK-AS64-CC1 %s
// CHECK-AS64-NOT: warning:
// CHECK-AS64-CC1: "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-AS64: "{{.*}}as{{(.exe)?}}" 
// CHECK-AS64: "-a64" 
// CHECK-AS64: "-many"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -Xassembler <arg> option. 
// RUN: %clang %s -### -c -fno-integrated-as 2>&1 \
// RUN:         -Xassembler -w \
// RUN:         --target=powerpc-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS32-Xassembler %s
// CHECK-AS32-Xassembler-NOT: warning:
// CHECK-AS32-Xassembler: "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-AS32-Xassembler: "{{.*}}as{{(.exe)?}}" 
// CHECK-AS32-Xassembler: "-a32" 
// CHECK-AS32-Xassembler: "-many"
// CHECK-AS32-Xassembler: "-w"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -Wa,<arg>,<arg> option.
// RUN: %clang %s -### -c -fno-integrated-as 2>&1 \
// RUN:         -Wa,-v,-w \
// RUN:         --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS64-Wa %s
// CHECK-AS64-Wa-NOT: warning:
// CHECK-AS64-Wa: "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-AS64-Wa: "{{.*}}as{{(.exe)?}}" 
// CHECK-AS64-Wa: "-a64" 
// CHECK-AS64-Wa: "-many"
// CHECK-AS64-Wa: "-v"
// CHECK-AS64-Wa: "-w"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Multiple input files.
// RUN: %clang -### -fno-integrated-as -c \
// RUN:         %S/Inputs/aix_ppc_tree/dummy0.s \
// RUN:         %S/Inputs/aix_ppc_tree/dummy1.s \
// RUN:         %S/Inputs/aix_ppc_tree/dummy2.s 2>&1 \
// RUN:         --target=powerpc-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-AS32-MultiInput %s
// CHECK-AS32-MultiInput-NOT: warning:
// CHECK-AS32-MultiInput: "{{.*}}as{{(.exe)?}}"
// CHECK-AS32-MultiInput: "-a32"
// CHECK-AS32-MultiInput: "-many"
// CHECK-AS32-MultiInput: "{{.*}}as{{(.exe)?}}"
// CHECK-AS32-MultiInput: "-a32"
// CHECK-AS32-MultiInput: "-many"
// CHECK-AS32-MultiInput: "{{.*}}as{{(.exe)?}}"
// CHECK-AS32-MultiInput: "-a32"
// CHECK-AS32-MultiInput: "-many"

// Check not passing no-integrated-as flag by default.
// RUN: %clang %s -### -c 2>&1 --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-IAS --implicit-check-not=-no-integrated-as %s
// CHECK-IAS-NOT: "-a64"
// CHECK-IAS: InstalledDir

// Check passing no-integrated-as flag if specified by user.
// RUN: %clang %s -### -c 2>&1 --target=powerpc64-ibm-aix7.1.0.0 -fno-integrated-as \
// RUN:   | FileCheck --check-prefix=CHECK-NOIAS %s
// CHECK-NOIAS: InstalledDir
// CHECK-NOIAS: -no-integrated-as
// CHECK-NOIAS: "-a64"
