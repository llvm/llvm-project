// RUN: %clang_cc1  -fsyntax-only -Wselector -verify -Wno-objc-root-class %s
// rdar://8851684

@interface Foo
- (void) foo;
- (void) bar;
@end

@implementation Foo
- (void) bar
{
}

- (void) foo
{
  SEL a,b,c;
  a = @selector(b1ar);  // expected-warning {{unimplemented selector 'b1ar'}}
  b = @selector(bar);
}
@end

@interface I
- length;
@end

SEL func()
{
    return  @selector(length);  // expected-warning {{unimplemented selector 'length'}}
}

// rdar://9545564
@class MSPauseManager;

@protocol MSPauseManagerDelegate 
@optional
- (void)pauseManagerDidPause:(MSPauseManager *)manager;
- (int)respondsToSelector:(SEL)aSelector;
@end

@interface MSPauseManager
{
  id<MSPauseManagerDelegate> _delegate;
}
@end


@implementation MSPauseManager
- (id) Meth {
  if ([_delegate respondsToSelector:@selector(pauseManagerDidPause:)])
    return 0;
  return 0;
}
@end

