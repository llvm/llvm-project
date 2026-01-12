// RUN:   %clang -### --target=x86_64-unknown-linux-llvm --sysroot=%S/Inputs/basic_llvm_linux_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_llvm_linux_tree/bin %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS
// CHECK-HEADERS: "-cc1"{{.*}}"-isysroot"{{.*}}"-internal-isystem" "{{.*}}include/x86_64-unknown-linux-llvm"
