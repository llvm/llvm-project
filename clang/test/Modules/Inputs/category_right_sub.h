@interface Foo(RightSub) <P2>
@property id right_sub_prop;
@end

@interface Foo() {
@public
  int right_sub_ivar;
}
@end

@protocol P4
- (void)p4_method;
@property (retain) id p4_prop;
@end

@interface Foo(LeftP4) <P4>
@end

// A hidden extension
@interface Foo ()
@property (assign) int hiddenPropertyFromExtension;
@end
