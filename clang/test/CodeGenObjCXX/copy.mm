// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

// rdar://problem/9158302
// This should not use a memmove_collectable in non-GC mode.
namespace test0 {
  struct A {
    id x;
  };

  // CHECK:    define{{.*}} ptr @_ZN5test04testENS_1AE(
  // CHECK:      alloca
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm(
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(
  // CHECK-NEXT: ret
  A *test(A a) {
    return new A(a);
  }
}


// rdar://9780211
@protocol bork
@end

namespace test1 {
template<typename T> struct RetainPtr {
  RetainPtr() {}
};


RetainPtr<id<bork> > x;
RetainPtr<id> y;

}
