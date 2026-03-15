// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -x c++ -fmodules-cache-path=%t -I %S/Inputs/glob-delete-with-virtual-dtor -emit-llvm -triple=i386-pc-win32 -o - | FileCheck %s --check-prefixes CHECK,CHECK32
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -x c++ -fmodules-cache-path=%t -I %S/Inputs/glob-delete-with-virtual-dtor -emit-llvm -triple=x86_64-pc-win32 -o - | FileCheck %s --check-prefixes CHECK,CHECK64

#include "glob-delete-with-virtual-dtor.h"

static void call_in_module_function(void) {
    in_h_tests();
}

void out_of_module_tests() {
  S* s = new S();
  ::delete s;
}

// CHECK32:      define {{.*}} @"??_GH@@UAEPAXI@Z"
// CHECK64:      define {{.*}} @"??_GH@@UEAAPEAXI@Z"
// CHECK:        store i32 %should_call_delete, ptr %[[SHOULD_DELETE_VAR:[0-9a-z._]+]], align 4
// CHECK:        store ptr %{{.*}}, ptr %[[RETVAL:retval]]
// CHECK:        %[[SHOULD_DELETE_VALUE:[0-9a-z._]+]] = load i32, ptr %[[SHOULD_DELETE_VAR]]
// CHECK32:        call x86_thiscallcc void @"??1H@@UAE@XZ"(ptr {{[^,]*}} %[[THIS:[0-9a-z]+]])
// CHECK64:        call void @"??1H@@UEAA@XZ"(ptr {{[^,]*}} %[[THIS:[0-9a-z]+]])
// CHECK-NEXT:   %[[AND:[0-9]+]] = and i32 %[[SHOULD_DELETE_VALUE]], 1
// CHECK-NEXT:   %[[CONDITION:[0-9]+]] = icmp eq i32 %[[AND]], 0
// CHECK-NEXT:   br i1 %[[CONDITION]], label %[[CONTINUE_LABEL:[0-9a-z._]+]], label %[[CALL_DELETE_LABEL:[0-9a-z._]+]]
//
// CHECK:      [[CALL_DELETE_LABEL]]
// CHECK-NEXT:   %[[AND:[0-9]+]] = and i32 %[[SHOULD_DELETE_VALUE]], 4
// CHECK-NEXT:   %[[CONDITION1:[0-9]+]] = icmp eq i32 %[[AND]], 0
// CHECK-NEXT:   br i1 %[[CONDITION1]], label %[[CALL_CLASS_DELETE:[0-9a-z._]+]], label %[[CALL_GLOB_DELETE:[0-9a-z._]+]]
//
// CHECK:      [[CALL_GLOB_DELETE]]
// CHECK32-NEXT:   call void @"??3@YAXPAXI@Z"
// CHECK64-NEXT:   call void @"??3@YAXPEAX_K@Z"
// CHECK-NEXT:   br label %[[CONTINUE_LABEL]]
//
// CHECK:      [[CALL_CLASS_DELETE]]
// CHECK32-NEXT:   call void @"??3H@@CAXPAX@Z"
// CHECK64-NEXT:   call void @"??3H@@CAXPEAX@Z"
// CHECK-NEXT:   br label %[[CONTINUE_LABEL]]
//
// CHECK:      [[CONTINUE_LABEL]]
// CHECK-NEXT:   %[[RET:.*]] = load ptr, ptr %[[RETVAL]]
// CHECK-NEXT:   ret ptr %[[RET]]
