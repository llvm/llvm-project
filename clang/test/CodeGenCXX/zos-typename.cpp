// RUN: %clang -S -emit-llvm -target s390x-none-zos -I%S -fexec-charset=UTF-8 %s -o - | FileCheck %s
// RUN: %clang -S -emit-llvm -target s390x-none-zos -I%S %s -o - | FileCheck %s
// RUN: %clang -S -emit-llvm -target s390x-none-zos -I%S -m32 %s -o - | FileCheck %s

#include <typeinfo>

class TestClass {};
struct TestStruct {};

const char *A = typeid(TestClass).name();
const char *B = typeid(TestStruct).name();

// CHECK: @_ZTS9TestClass = {{.*}} c"\F9\E3\85\A2\A3\C3\93\81\A2\A2\009TestClass\00"
// CHECK: @_ZTS10TestStruct = {{.*}} c"\F1\F0\E3\85\A2\A3\E2\A3\99\A4\83\A3\0010TestStruct\00"
