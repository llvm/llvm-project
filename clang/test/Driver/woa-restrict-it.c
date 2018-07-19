// RUN: %clang -target armv7-windows -### %s 2>&1 | FileCheck %s

// CHECK: "-backend-option" "-arm-restrict-it"

