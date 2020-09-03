// Tests that the AIX toolchain adds system includes to its search path.

// Check powerpc-ibm-aix, 32-bit/64-bit.
// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefix=CHECK-INTERNAL-INCLUDE %s

// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefix=CHECK-INTERNAL-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefix=CHECK-INTERNAL-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefix=CHECK-INTERNAL-INCLUDE %s

// CHECK-INTERNAL-INCLUDE:	{{.*}}clang{{.*}}" "-cc1"
// CHECK-INTERNAL-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-INTERNAL-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-INTERNAL-INCLUDE:	"-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-INTERNAL-INCLUDE:	"-internal-isystem" "[[SYSROOT]]/usr/include"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nostdinc option.
// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// CHECK-NOSTDINC-INCLUDE:	{{.*}}clang{{.*}}" "-cc1"
// CHECK-NOSTDINC-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDINC-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOSTDINC-INCLUDE-NOT:	"-internal-isystem"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nostdlibinc option.
// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// CHECK-NOSTDLIBINC-INCLUDE:	{{.*}}clang{{.*}}" "-cc1"
// CHECK-NOSTDLIBINC-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDLIBINC-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOSTDLIBINC-INCLUDE:	"-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-NOSTDLIBINC-INCLUDE-NOT:	"-internal-isystem" "[[SYSROOT]]/usr/include"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nobuiltininc option.
// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefix=CHECK-NOBUILTININC-INCLUDE %s

// RUN: %clangxx -### -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefix=CHECK-NOBUILTININC-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefix=CHECK-NOBUILTININC-INCLUDE %s

// RUN: %clang -### -xc -no-canonical-prefixes %s 2>&1 \
// RUN:		-target powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefix=CHECK-NOBUILTININC-INCLUDE %s

// CHECK-NOBUILTININC-INCLUDE:	{{.*}}clang{{.*}}" "-cc1"
// CHECK-NOBUILTININC-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOBUILTININC-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOBUILTININC-INCLUDE-NOT:	"-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-NOBUILTININC-INCLUDE:	"-internal-isystem" "[[SYSROOT]]/usr/include"
