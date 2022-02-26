// Tests that the AIX toolchain adds system includes to its search path.

// Check powerpc-ibm-aix, 32-bit/64-bit.
// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefixes=CHECK-INTERNAL-INCLUDE,CHECK-INTERNAL-INCLUDE-CXX %s

// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefixes=CHECK-INTERNAL-INCLUDE,CHECK-INTERNAL-INCLUDE-CXX %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefix=CHECK-INTERNAL-INCLUDE %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:   | FileCheck -check-prefix=CHECK-INTERNAL-INCLUDE %s

// CHECK-INTERNAL-INCLUDE:      "-cc1"
// CHECK-INTERNAL-INCLUDE:      "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-INTERNAL-INCLUDE:      "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-INTERNAL-INCLUDE-CXX:  "-internal-isystem" "[[SYSROOT]]{{(/|\\\\)}}opt{{(/|\\\\)}}IBM{{(/|\\\\)}}openxlCSDK{{(/|\\\\)}}include{{(/|\\\\)}}c++{{(/|\\\\)}}v1"
// CHECK-INTERNAL-INCLUDE-CXX:  "-D__LIBC_NO_CPP_MATH_OVERLOADS__"
// CHECK-INTERNAL-INCLUDE:      "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-INTERNAL-INCLUDE:      "-internal-isystem" "[[SYSROOT]]/usr/include"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nostdinc option.
// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINC-INCLUDE %s

// CHECK-NOSTDINC-INCLUDE:	"-cc1"
// CHECK-NOSTDINC-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDINC-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOSTDINC-INCLUDE-NOT:	"-internal-isystem"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nostdlibinc option.
// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nostdlibinc \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDLIBINC-INCLUDE %s

// CHECK-NOSTDLIBINC-INCLUDE:	"-cc1"
// CHECK-NOSTDLIBINC-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDLIBINC-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOSTDLIBINC-INCLUDE:	"-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-NOSTDLIBINC-INCLUDE-NOT:  "-internal-isystem" "[[SYSROOT]]{{(/|\\\\)}}opt{{(/|\\\\)}}IBM{{(/|\\\\)}}openxlCSDK{{(/|\\\\)}}include{{(/|\\\\)}}c++{{(/|\\\\)}}v1"
// CHECK-NOSTDLIBINC-INCLUDE-NOT:  "-D__LIBC_NO_CPP_MATH_OVERLOADS__"
// CHECK-NOSTDLIBINC-INCLUDE-NOT:	"-internal-isystem" "[[SYSROOT]]/usr/include"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nobuiltininc option.
// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefixes=CHECK-NOBUILTININC-INCLUDE,CHECK-NOBUILTININC-INCLUDE-CXX %s

// RUN: %clangxx -### %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefixes=CHECK-NOBUILTININC-INCLUDE,CHECK-NOBUILTININC-INCLUDE-CXX  %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefix=CHECK-NOBUILTININC-INCLUDE %s

// RUN: %clang -### -xc %s 2>&1 \
// RUN:		--target=powerpc64-ibm-aix \
// RUN:		-resource-dir=%S/Inputs/resource_dir \
// RUN:		--sysroot=%S/Inputs/basic_aix_tree \
// RUN:		-nobuiltininc \
// RUN:   | FileCheck -check-prefix=CHECK-NOBUILTININC-INCLUDE %s

// CHECK-NOBUILTININC-INCLUDE:	"-cc1"
// CHECK-NOBUILTININC-INCLUDE:	"-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOBUILTININC-INCLUDE:	"-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOBUILTININC-INCLUDE-NOT:	"-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-NOBUILTININC-INCLUDE-CXX:  "-internal-isystem" "[[SYSROOT]]{{(/|\\\\)}}opt{{(/|\\\\)}}IBM{{(/|\\\\)}}openxlCSDK{{(/|\\\\)}}include{{(/|\\\\)}}c++{{(/|\\\\)}}v1"
// CHECK-NOBUILTININC-INCLUDE-CXX:  "-D__LIBC_NO_CPP_MATH_OVERLOADS__"
// CHECK-NOBUILTININC-INCLUDE:	"-internal-isystem" "[[SYSROOT]]/usr/include"

// Check powerpc-ibm-aix, 32-bit/64-bit. -nostdinc++ option.
// RUN: %clangxx -### %s 2>&1 \
// RUN:  --target=powerpc-ibm-aix \
// RUN:  -resource-dir=%S/Inputs/resource_dir \
// RUN:  --sysroot=%S/Inputs/basic_aix_tree \
// RUN:  -nostdinc++ \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINCXX-INCLUDE %s

// RUN: %clangxx -### %s 2>&1 \
// RUN:  --target=powerpc64-ibm-aix \
// RUN:  -resource-dir=%S/Inputs/resource_dir \
// RUN:  --sysroot=%S/Inputs/basic_aix_tree \
// RUN:  -nostdinc++ \
// RUN:   | FileCheck -check-prefix=CHECK-NOSTDINCXX-INCLUDE  %s

// CHECK-NOSTDINCXX-INCLUDE:      "-cc1"
// CHECK-NOSTDINCXX-INCLUDE:      "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDINCXX-INCLUDE:      "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-NOSTDINCXX-INCLUDE:      "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-NOSTDINCXX-INCLUDE-NOT:  "-internal-isystem" "[[SYSROOT]]{{(/|\\\\)}}opt{{(/|\\\\)}}IBM{{(/|\\\\)}}openxlCSDK{{(/|\\\\)}}include{{(/|\\\\)}}c++{{(/|\\\\)}}v1"
// CHECK-NOSTDINCXX-INCLUDE-NOT:  "-D__LIBC_NO_CPP_MATH_OVERLOADS__"
// CHECK-NOSTDINCXX-INCLUDE:      "-internal-isystem" "[[SYSROOT]]/usr/include"

// Check powerpc-ibm-aix, 32-bit. -stdlib=libstdc++ invokes fatal error.
// RUN: not --crash %clangxx %s 2>&1 -### \
// RUN:        --target=powerpc-ibm-aix \
// RUN:        -stdlib=libstdc++ \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-INCLUDE-LIBSTDCXX %s

// Check powerpc64-ibm-aix, 64-bit. -stdlib=libstdc++ invokes fatal error.
// RUN: not --crash %clangxx %s 2>&1 -### \
// RUN:        --target=powerpc64-ibm-aix \
// RUN:        -stdlib=libstdc++ \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-INCLUDE-LIBSTDCXX %s

// CHECK-INCLUDE-LIBSTDCXX: LLVM ERROR: picking up libstdc++ headers is unimplemented on AIX
