// RUN: %clang -### -fopenmp=libomp --target=x86_64-linux-gnu -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin %s 2>&1 | FileCheck %s

void foo() {}

// CHECK: -L{{.*}}Inputs{{.*}}basic_linux_tree
