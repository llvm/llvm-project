/* For use with the objc_import.m test */

@interface TestPCH
+ alloc;
- (void)instMethod;
@end

@class NewID1;
@compatibility_alias OldID1 NewID1;
@class OldID1;
@class OldID1;

@class NewID2;
@compatibility_alias OldID2 NewID2;
@class OldID2;
@interface OldID2
-(void)meth;
@end
