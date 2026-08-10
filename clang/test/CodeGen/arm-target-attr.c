// RUN: %clang_cc1 -triple thumb-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix CHECKPOS %s
// RUN: %clang_cc1 -triple thumb-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix CHECKNEG %s
// RUN: %clang_cc1 -triple arm-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix CHECKPOS %s
// RUN: %clang_cc1 -triple arm-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix CHECKNEG %s

__attribute__((target("arm"))) void test_target_arm(void) {
  // CHECKPOS: define{{.*}} void @test_target_arm() [[ARM_ATTRS:#[0-9]+]]
  // CHECKNEG: define{{.*}} void @test_target_arm() [[ARM_ATTRS:#[0-9]+]]
}

__attribute__((target("thumb"))) void test_target_thumb(void) {
  // CHECKPOS: define{{.*}} void @test_target_thumb() [[THUMB_ATTRS:#[0-9]+]]
  // CHECKNEG: define{{.*}} void @test_target_thumb() [[THUMB_ATTRS:#[0-9]+]]
}

// CHECKPOS: attributes [[ARM_ATTRS]] = { {{.*}} "target-features"="{{.*}}-thumb-mode{{.*}}"
// CHECKPOS: attributes [[THUMB_ATTRS]] = { {{.*}} "target-features"="{{.*}}+thumb-mode{{.*}}"
// CHECKNEG-NOT: attributes [[ARM_ATTRS]] = { {{.*}} "target-features"="{{.*}}+thumb-mode{{.*}}"
// CHECKNEG-NOT: attributes [[THUMB_ATTRS]] = { {{.*}} "target-features"="{{.*}}-thumb-mode{{.*}}"
