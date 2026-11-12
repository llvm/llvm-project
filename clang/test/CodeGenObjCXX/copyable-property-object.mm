// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

struct POD {
  int array[3][4];
  id objc_obj;
};

struct D  { 
  POD pod_array[2][3];
};

@interface I
{
  D Property1;
}
@property D Property1;
- (D) val;
- (void) set : (D) d1;
@end

@implementation I
@synthesize Property1;
- (D) val { return Property1; }
- (void) set : (D) d1 { Property1 = d1; }
@end
// CHECK: {{call.*@objc_memmove_collectable}}
// CHECK: {{call.*@objc_memmove_collectable}}

