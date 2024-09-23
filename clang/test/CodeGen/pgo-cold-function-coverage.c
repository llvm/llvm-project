// Test -fprofile-generate-cold-function-coverage 
// RUN: %clang -O2 -fprofile-generate-cold-function-coverage=/xxx/yyy/ -fprofile-sample-accurate -fprofile-sample-use=%S/Inputs/pgo-cold-func.prof  -S -emit-llvm -o - %s | FileCheck %s

// CHECK: @__llvm_profile_filename = {{.*}} c"/xxx/yyy/default_%m.profraw\00"

// CHECK: @__profc_bar
// CHECK-NOT: @__profc_foo

int bar(int x) { return x;}
int foo(int x) { 
    return x;
}
