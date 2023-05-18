/* For use with the methods.m test */

@interface A
@end

@interface B
@end

@interface TestPCH
- (void)instMethod;
@end

@implementation TestPCH
- (void)instMethod {
  @try {
  } @catch(A *a) {
  } @catch(B *b) {
  } @catch(...) {
  } @finally {
  }
}
@end
