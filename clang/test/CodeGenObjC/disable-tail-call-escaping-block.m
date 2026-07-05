// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -fno-escaping-block-tail-calls -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define{{.*}} void @test(
// CHECK: store ptr @[[TEST_BLOCK_INVOKE0:.*invoke.*]], ptr
// CHECK: store ptr @[[TEST_BLOCK_INVOKE1:.*invoke.*]], ptr
// CHECK: store ptr @[[TEST_BLOCK_INVOKE2:.*invoke.*]], ptr
// CHECK: store ptr @[[TEST_BLOCK_INVOKE3:.*invoke.*]], ptr
// CHECK: store ptr @[[TEST_BLOCK_INVOKE4:.*invoke.*]], ptr
// CHECK: store ptr @[[TEST_BLOCK_INVOKE5:.*invoke.*]], ptr
// CHECK: store ptr @[[TEST_BLOCK_INVOKE6:.*invoke.*]], ptr

// CHECK: define internal void @[[TEST_BLOCK_INVOKE0]]({{.*}}) #[[DISABLEATTR:.*]] {
// CHECK: define internal void @[[TEST_BLOCK_INVOKE1]]({{.*}}) #[[ENABLEATTR:.*]] {
// CHECK: define internal void @[[TEST_BLOCK_INVOKE2]]({{.*}}) #[[DISABLEATTR]] {
// CHECK: define internal void @[[TEST_BLOCK_INVOKE3]]({{.*}}) #[[DISABLEATTR]] {
// CHECK: define internal void @[[TEST_BLOCK_INVOKE4]]({{.*}}) #[[ENABLEATTR]] {
// CHECK: define internal void @[[TEST_BLOCK_INVOKE5]]({{.*}}) #[[DISABLEATTR]] {
// CHECK: define internal void @[[TEST_BLOCK_INVOKE6]]({{.*}}) #[[ENABLEATTR]] {

// CHECK-NOT: attributes #[[ENABLEATTR]] = {{{.*}}"disable-tail-calls"="false"{{.*}}}
// CHECK: attributes #[[DISABLEATTR]] = {{{.*}}"disable-tail-calls"="true"{{.*}}}

typedef void (^BlockTy)(void);
typedef void (*NoEscapeFnTy)(__attribute__((noescape)) BlockTy);

void callee0(__attribute__((noescape)) BlockTy);
void callee1(BlockTy);

__attribute__((objc_root_class))
@interface C0
-(void)m0:(__attribute__((noescape)) BlockTy)p;
-(void)m1:(BlockTy)p;
@end

@implementation C0
-(void)m0:(__attribute__((noescape)) BlockTy)p {}
-(void)m1:(BlockTy)p {}
@end

NoEscapeFnTy noescapefunc;

void test(id a, C0 *c0) {
  BlockTy b0 = ^{ (void)a; }; // disable tail-call optimization.
  callee0(b0);
  callee0(^{ (void)a; }); // enable tail-call optimization.
  callee1(^{ (void)a; }); // disable tail-call optimization.

  BlockTy b1 = ^{ (void)a; }; // disable tail-call optimization.
  [c0 m0:b1];
  [c0 m0:^{ (void)a; }]; // enable tail-call optimization.
  [c0 m1:^{ (void)a; }]; // disable tail-call optimization.

  noescapefunc(^{ (void)a; }); // enable tail-call optimization.
}
