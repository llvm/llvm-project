// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -g -emit-llvm -o - %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -g -emit-llvm -o - %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -g -emit-llvm -o - %s
// RUN: %clang_cc1 -x objective-c++ -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -g -emit-llvm -o - %s

// rdar://7252252
@interface Loop {
@public
  __weak Loop *_loop;
}
@end

@implementation Loop @end

void loop(Loop *L) {
  L->_loop = 0;
}
