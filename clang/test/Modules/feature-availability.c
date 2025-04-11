// RUN: %clang_cc1 -triple arm64-apple-macosx -fmodules %S/Inputs/feature-availability/module.modulemap -fmodule-name=Feature1 -emit-module -o %t/feature1.pcm
// RUN: %clang_cc1 -triple arm64-apple-macosx -fmodules %S/Inputs/feature-availability/module.modulemap -fmodule-name=Feature2 -fmodule-file=%t/feature1.pcm -emit-module -o %t/feature2.pcm
// RUN: %clang_cc1 -triple arm64-apple-macosx -fmodules -fmodule-file=%t/feature2.pcm -I %S/Inputs/feature-availability -emit-llvm -o - %s | FileCheck %s

#include <feature-availability.h>
#include "feature2.h"

#define AVAIL 0
#define UNAVAIL 1

int pred1(void);
static struct __AvailabilityDomain feature3 __attribute__((availability_domain(feature3))) = {__AVAILABILITY_DOMAIN_DYNAMIC, pred1};

void func0(void);
__attribute__((availability(domain:feature1, AVAIL))) void func1(void);
__attribute__((availability(domain:feature2, UNAVAIL))) void func2(void);
__attribute__((availability(domain:feature3, AVAIL))) void func3(void);

// CHECK-LABEL: define void @test1()
// CHECK-NOT: call
// CHECK: call void @func1()
// CHECK-NEXT: ret void

void test1(void) {
  if (__builtin_available(domain:feature1))
    func1();
  else
    func0();
}

// CHECK: define void @test2()
// CHECK-NOT: call
// CHECK: call void @func2()
// CHECK-NEXT: ret void

void test2(void) {
  if (__builtin_available(domain:feature2))
    func0();
  else
    func2();
}

// CHECK: define void @test3()
// CHECK: %[[CALL:.*]] = call i32 @pred1()
// CHECK-NEXT: %[[TOBOOL:.*]] = icmp ne i32 %[[CALL]], 0
// CHECK-NEXT: br i1 %[[TOBOOL]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
//
// CHECK: [[IF_THEN]]:
// CHECK-NEXT: call void @func3()
// CHECK-NEXT: br label %[[IF_END:.*]]
//
// CHECK: [[IF_ELSE]]:
// CHECK-NEXT: call void @func0()
// CHECK-NEXT: br label %[[IF_END]]
//
// CHECK: [[IF_END]]:
// CHECK-NEXT: ret void

void test3(void) {
  if (__builtin_available(domain:feature3))
    func3();
  else
    func0();
}
