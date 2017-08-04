#include "objcHeader.h"

@implementation MyClass

#ifdef MIX_IMPL
+ (void)classMethod { }

- (void)method:(int)x with:(int)y { }
#endif

@end
// CHECK1: "{{.*}}objcClass.m" "- (void)method { \n  <#code#>;\n}\n\n+ (void)classMethod { \n  <#code#>;\n}\n\n- (void)implementedMethod { \n  <#code#>;\n}\n\n- (void)method:(int)x with:(int)y { \n  <#code#>;\n}\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// CHECK2: "{{.*}}objcClass.m" "- (void)method { \n  <#code#>;\n}\n\n- (void)implementedMethod { \n  <#code#>;\n}\n\n" [[@LINE-2]]:1

// CHECK-CAT-NO-IMPL: "{{.*}}objcClass.m" "- (void)thisCategoryMethodShouldBeInTheClassImplementation { \n  <#code#>;\n}\n\n" 11:1 -> 11:1
