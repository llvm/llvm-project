// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc-only -emit-llvm -o %t %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc-only -emit-llvm -o %t %s

@interface I0 {
  I0 *_f0;
}
@property (retain) id p0;
@end 

@implementation I0 
  @synthesize p0 = _f0;
@end

