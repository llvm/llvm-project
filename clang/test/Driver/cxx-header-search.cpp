// Check that the paths of C++ headers vendored in clang 
// will be found before the C++ standard libraries

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX %s
// CHECK-LIBCXX: "-cc1"
// CHECK-LIBCXX-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LIBCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]][[SEP:/|\\\\]]include[[SEP]]cxx"
// CHECK-LIBCXX-SAME: "-internal-isystem" "{{[^"]+}}[[SEP]]c++[[SEP]]v1"
// CHECK-LIBCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]][[SEP]]include"

// RUN: %clang -### %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     -stdlib++-isystem %S/Inputs/stdlib \
// RUN:   | FileCheck --check-prefix=CHECK-STDLIBCXX %s
// CHECK-STDLIBCXX: "-cc1"
// CHECK-STDLIBCXX-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-STDLIBCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]][[SEP:/|\\\\]]include[[SEP]]cxx"
// CHECK-STDLIBCXX-SAME: "-internal-isystem" "{{[^"]+}}[[SEP]]Inputs[[SEP]]stdlib"
// CHECK-STDLIBCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]][[SEP]]include"

// The directory is not added with -nostdinc or -nobuiltininc, or for C.
// RUN: %clang -### %s -fsyntax-only -nostdinc 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### %s -fsyntax-only -nobuiltininc 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -x c %s -fsyntax-only 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// CHECK-NONE: "-cc1"
// CHECK-NONE-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NONE-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include{{/|\\\\}}cxx"
