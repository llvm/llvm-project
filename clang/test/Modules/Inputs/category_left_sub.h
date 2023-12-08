@interface Foo(LeftSub) <P1>
- (void)left_sub;
@end

@protocol P3 
- (void)p3_method;
@property (retain) id p3_prop;
@end

@interface Foo(LeftP3) <P3>
@end
