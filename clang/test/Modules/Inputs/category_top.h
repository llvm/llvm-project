@interface Foo
@end

@interface Foo(Top)
-(void)top;
@end

@interface Foo(Top2)
-(void)top2;
@end

@interface Foo(Top3)
-(void)top3;
@end

@protocol P1
@end

@protocol P2
@end

@protocol P3, P4;

