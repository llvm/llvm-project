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
// HASSTRET: [[T0:%.*]] = call i8* (i8*, i8*, ...)* @objc_msg_lookup_stret(i8* bitcast (i64* @_OBJC_CLASS_Test0 to i8*),
// HASSTRET-NEXT: [[T1:%.*]] = bitcast i8* (i8*, i8*, ...)* [[T0]] to void (%struct.test*, i8*, i8*)*
// HASSTRET-NEXT: call void [[T1]](%struct.test* sret(%struct.test) {{.*}}, i8* bitcast (i64* @_OBJC_CLASS_Test0 to i8*),

// NOSTRET-LABEL: define{{.*}} void @test0()
// NOSTRET: [[T0:%.*]] = call i8* (i8*, i8*, ...)* @objc_msg_lookup(i8*
// NOSTRET-NEXT: [[T1:%.*]] = bitcast i8* (i8*, i8*, ...)* [[T0]] to void (%struct.test*, i8*, i8*)*
// NOSTRET-NEXT: call void [[T1]](%struct.test* sret(%struct.test) {{.*}}, i8* {{.*}}, i8* bitcast ([2 x { i8*, i8* }]*
