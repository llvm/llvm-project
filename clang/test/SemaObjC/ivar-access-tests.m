// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface MySuperClass
{
@private
  int private;

@protected
  int protected;

@public
  int public;
}
@end

@implementation MySuperClass
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private;   
    access = s->protected;
}
@end


@interface MyClass : MySuperClass 
@end

@implementation MyClass
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private; // expected-error {{instance variable 'private' is private}}
    access = s->protected;
    MyClass *m=0;
    access = m->private; // expected-error {{instance variable 'private' is private}}
    access = m->protected;
}
@end


@interface Deeper : MyClass
@end

@implementation Deeper 
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private; // expected-error {{instance variable 'private' is private}}
    access = s->protected;
    MyClass *m=0;
    access = m->private; // expected-error {{instance variable 'private' is private}}
    access = m->protected;
}
@end

@interface Unrelated
@end

@implementation Unrelated 
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private; // expected-error {{instance variable 'private' is private}}
    access = s->protected; // expected-error {{instance variable 'protected' is protected}}
    MyClass *m=0;
    access = m->private; // expected-error {{instance variable 'private' is private}}
    access = m->protected; // expected-error {{instance variable 'protected' is protected}}
}
@end

int main (void)
{
  MySuperClass *s = 0;
  int access;
  access = s->private;   // expected-error {{instance variable 'private' is private}}
  access = s->protected; // expected-error {{instance variable 'protected' is protected}}
  return 0;
}

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end 
@interface NSObject <NSObject> {}
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSResponder : NSObject <NSCoding> {}
@end 
@protocol NSAnimatablePropertyContainer
- (id)animator;
@end
extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
  struct __VFlags2 {
  }
  _vFlags2;
}
@end
@class NSFontDescriptor, NSAffineTransform, NSGraphicsContext;
@interface NSScrollView : NSView {}
@end

@class CasperMixerView;
@interface CasperDiffScrollView : NSScrollView {
@private
  CasperMixerView *_comparatorView;
  NSView *someField;
}
@end

@implementation CasperDiffScrollView
+ (void)initialize {}
static void _CasperDiffScrollViewInstallMixerView(CasperDiffScrollView *scrollView) {
  if (scrollView->someField != ((void *)0)) {
  }
}
@end
