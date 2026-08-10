// RUN: %clang %s -target x86_64-apple-driverkit19.0 -### 2>&1 | FileCheck %s

int main() { return 0; }
// CHECK: "-target-cpu" "nehalem"
