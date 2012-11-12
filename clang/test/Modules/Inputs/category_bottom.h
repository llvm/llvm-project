@__experimental_modules_import category_left;

@interface Foo(Bottom)
-(void)bottom;
@end

@__experimental_modules_import category_right;

@interface LeftFoo(Bottom)
-(void)bottom;
@end
