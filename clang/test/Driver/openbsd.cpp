// RUN: %clangxx %s -### -o %t.o -target amd64-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// RUN: %clangxx %s -### -o %t.o -target i686-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// RUN: %clangxx %s -### -o %t.o -target aarch64-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// RUN: %clangxx %s -### -o %t.o -target arm-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CXX %s
// CHECK-CXX: "-lc++" "-lc++abi" "-lpthread" "-lm"

// RUN: %clangxx %s -### -pg -o %t.o -target amd64-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// RUN: %clangxx %s -### -pg -o %t.o -target i686-pc-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// RUN: %clangxx %s -### -pg -o %t.o -target aarch64-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// RUN: %clangxx %s -### -pg -o %t.o -target arm-unknown-openbsd 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG-CXX %s
// CHECK-PG-CXX: "-lc++_p" "-lc++abi_p" "-lpthread_p" "-lm_p"
