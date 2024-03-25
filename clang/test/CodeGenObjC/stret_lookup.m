// RUN: %clang_cc1 -DSTRET -triple x86_64-pc-linux-gnu -fobjc-runtime=objfw -emit-llvm -o - %s | FileCheck -check-prefix=HASSTRET %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fobjc-runtime=gcc -emit-llvm -o - %s | FileCheck -check-prefix=NOSTRET %s

// Test stret lookup

struct test {
  char test[1024];
};
@interface Test0
+ (struct test)test;
@end
void test0(void) {
  struct test t;
#if (defined(STRET) && defined(__OBJFW_RUNTIME_ABI__)) || \
    (!defined(STRET) && !defined(__OBJFW_RUNTIME_ABI__))
  t = [Test0 test];
#endif
  (void)t;
}

// HASSTRET-LABEL: define{{.*}} void @test0()
// HASSTRET: [[T0:%.*]] = call ptr @objc_msg_lookup_stret(ptr @_OBJC_CLASS_Test0,
// HASSTRET-NEXT: call void [[T0]](ptr dead_on_unwind writable sret(%struct.test) {{.*}}, ptr noundef @_OBJC_CLASS_Test0,

// NOSTRET-LABEL: define{{.*}} void @test0()
// NOSTRET: [[T0:%.*]] = call ptr @objc_msg_lookup(ptr
// NOSTRET-NEXT: call void [[T0]](ptr dead_on_unwind writable sret(%struct.test) {{.*}}, ptr {{.*}}, ptr noundef
