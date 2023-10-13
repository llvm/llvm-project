// Check the C++ header path (libstdc++)
// RUN: %clangxx --target=x86_64-unknown-haiku --stdlib=libstdc++ -### %s 2>&1 \
// RUN:   --sysroot=%S/Inputs/haiku_x86_64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LIBSTDCXX-HEADER-PATH %s
// CHECK-LIBSTDCXX-HEADER-PATH: "-internal-isystem" "[[SYSROOT:[^"]+]]/boot/system/develop/tools/lib/gcc/x86_64-unknown-haiku/13.2.0/../../../gcc/x86_64-unknown-haiku/13.2.0/include/c++/"
// CHECK-LIBSTDCXX-HEADER-PATH-SAME: "-internal-isystem" "[[SYSROOT:[^"]+]]/boot/system/develop/tools/lib/gcc/x86_64-unknown-haiku/13.2.0/../../../gcc/x86_64-unknown-haiku/13.2.0/include/c++//backward"

// Check the C++ header path (when using libc++)
// RUN: %clangxx --target=x86_64-unknown-haiku --stdlib=libc++ -### %s 2>&1 \
// RUN:   --sysroot=%S/Inputs/haiku_x86_64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-HEADER-PATH %s
// CHECK-LIBCXX-HEADER-PATH: "-internal-isystem" "[[SYSROOT:[^"]+]]/boot/system/develop/headers/c++/v1"
