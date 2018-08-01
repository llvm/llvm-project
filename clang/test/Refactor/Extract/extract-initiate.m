
@interface I
@end

@implementation I

- (int)computeAreaSquaredGivenWidth:(int)width height:(int)height {
  int area = width * height;
  return area * area;
}
//CHECK1: Initiated the 'extract' action at [[@LINE-3]]:14 -> [[@LINE-3]]:28
// RUN: clang-refactor-test initiate -action extract -selected=%s:8:14-8:28 %s | FileCheck --check-prefix=CHECK1 %s

@end
