// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-include-fixer -db=yaml -input=%p/Inputs/fake_yaml_db.yaml %t.cpp --
// RUN: FileCheck %s -input-file=%t.cpp

// CHECK-NOT: #include
// CHECK: doesnotexist f;

namespace b {
doesnotexist f;
}
