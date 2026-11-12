// RUN: %clang_cc1  -fsyntax-only -Wselector -verify -Wno-objc-root-class %s

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
  a = @selector(b1ar);
  b = @selector(bar);
}
@end

@interface I
- length;
@end

SEL func(void)
{
    return  @selector(length);  // expected-warning {{no method with selector 'length' is implemented in this translation unit}}
}

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

@class NSXPCConnection;

@interface NSObject
@end

@interface INTF : NSObject
{
  NSXPCConnection *cnx; // Comes in as a parameter.
}
- (void) Meth;
@end

extern SEL MySelector(SEL s);

@implementation INTF
- (void) Meth {
  if( [cnx respondsToSelector:MySelector(@selector( _setQueue: ))] )
  {
  }

  if( [cnx respondsToSelector:@selector( _setQueueXX: )] ) // No warning here.
  {
  }
  if( [cnx respondsToSelector:(@selector( _setQueueXX: ))] ) // No warning here.
  {
  }
}
@end

@interface UxTechTest : NSObject
- (int) invalidate : (id)Arg;
+ (int) C_invalidate : (int)arg;
@end

@interface UxTechTest(CAT)
- (char) invalidate : (int)arg;
+ (int) C_invalidate : (char)arg;
@end

@interface NSPort : NSObject
- (double) invalidate : (void*)Arg1;
+ (int) C_invalidate : (id*)arg;
@end


@interface USEText : NSPort
- (int) invalidate : (int)arg;
@end

@implementation USEText
- (int) invalidate :(int) arg { return 0; }
@end

@interface USETextSub : USEText
- (int) invalidate : (id)arg;
@end

@interface I16428638
- (int) compare: (I16428638 *) arg1; // commenting out this line avoids the warning
@end

@interface J16428638
- (int) compare: (J16428638 *) arg1;
@end

@implementation J16428638
- (void)method {
    SEL s = @selector(compare:); // spurious warning
    (void)s;
}
- (int) compare: (J16428638 *) arg1 {
    return 0;
}
@end

void test16428638(void) {
    SEL s = @selector(compare:);
    (void)s;
}

@class NSString;
@interface SELCanary : NSObject
@property (readonly, nonatomic) NSString *name;
@property (nonatomic, getter = isHidden) char hidden;
@property (nonatomic, copy, getter = hasFish, setter = setFish:) NSString *ridiculousFish;
@end

@implementation SELCanary
- (void) Meth {
 SEL properties[] = {
  @selector(name),
  @selector(isHidden),
  @selector(setHidden:),
  @selector(hasFish),
  @selector(setFish:)
 };
}
@end

