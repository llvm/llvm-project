// RUN: %clang -### -malign-double %s  2>&1 | FileCheck %s

// Make sure -malign-double is passed through the driver.

// CHECK: "-malign-double"
