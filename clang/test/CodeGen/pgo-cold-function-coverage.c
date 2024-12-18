// Test -fprofile-generate-cold-function-coverage 

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang --target=x86_64 -O2 -fprofile-generate-cold-function-coverage=/xxx/yyy/ -fprofile-sample-accurate -fprofile-sample-use=%t/pgo-cold-func.prof  -S -emit-llvm -o - %t/pgo-cold-func.c | FileCheck %s

// CHECK: @__llvm_profile_filename = {{.*}} c"/xxx/yyy/default_%m.profraw\00"

// CHECK: store i8 0, ptr @__profc_bar, align 1
// CHECK-NOT: @__profc_foo 

//--- pgo-cold-func.prof
foo:1:1
 1: 1

//--- pgo-cold-func.c
int bar(int x) { return x;}
int foo(int x) { 
    return x;
}
