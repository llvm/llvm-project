@import category_top;

@interface Foo(Left)
-(void)left;
@end

@interface LeftFoo
-(void)left;
@end

@interface Foo(Duplicate)
@end

@interface Foo(Duplicate)
@end
