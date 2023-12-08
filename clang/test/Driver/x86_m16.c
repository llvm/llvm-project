// RUN: %clang -target i386 -m16 -### -c %s 2>&1 | FileCheck %s

// CHECK: Target: i386-{{.*}}-{{.*}}-code16

