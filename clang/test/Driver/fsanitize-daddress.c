// RUN: %clang --target=x86_64-linux-gnu -fsanitize=daddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SUPPORTED
// RUN: %clang --target=aarch64-linux-gnu -fsanitize=daddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SUPPORTED
// RUN: %clang --target=powerpc64le-linux-gnu -fsanitize=daddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SUPPORTED
// CHECK-SUPPORTED-NOT: unsupported option

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib -fsanitize=daddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE
// CHECK-DEVICE-NOT: unsupported option

// RUN: not %clang --target=i386-pc-openbsd -fsanitize=daddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-OPENBSD
// CHECK-OPENBSD: unsupported option '-fsanitize=daddress' for target 'i386-pc-openbsd'

// RUN: not %clang --target=x86_64-apple-darwin -fsanitize=daddress %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN
// CHECK-DARWIN: unsupported option '-fsanitize=daddress' for target

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=daddress,address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ASAN
// CHECK-ASAN: error: invalid argument '-fsanitize=daddress' not allowed with '-fsanitize=address'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=daddress,thread %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TSAN
// CHECK-TSAN: error: invalid argument '-fsanitize=daddress' not allowed with '-fsanitize=thread'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=daddress,memory %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MSAN
// CHECK-MSAN: error: invalid argument '-fsanitize=daddress' not allowed with '-fsanitize=memory'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=daddress,undefined %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-UBSAN
// CHECK-UBSAN: error: invalid argument '-fsanitize=daddress' not allowed with '-fsanitize=undefined'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=daddress,realtime %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-RTSAN
// CHECK-RTSAN: error: invalid argument '-fsanitize=daddress' not allowed with '-fsanitize=realtime'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=daddress,type %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-TYSAN
// CHECK-TYSAN: error: invalid argument '-fsanitize=daddress' not allowed with '-fsanitize=type'

// The runtime has to be in the executable itself for its definitions to be the
// ones the offloading runtime binds to.
// RUN: %clang %s -### -o %t.o --target=x86_64-unknown-linux -fuse-ld=ld \
// RUN:     -fsanitize=daddress \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LINK
// CHECK-LINK: "--whole-archive" "{{.*}}libclang_rt.dasan{{[^.]*}}.a" "--no-whole-archive"
// CHECK-LINK: "-ldl"

int main(void) { return 0; }
