// RUN: %clangxx -### %s 2>&1 | FileCheck %s

// PR5803

// CHECK: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated
// CHECK: "-cc1" {{.*}} "-x" "c++"
