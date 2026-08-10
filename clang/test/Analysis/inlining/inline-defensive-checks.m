// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config suppress-inlined-defensive-checks=true -verify %s

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
  Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

// Check that inline defensive checks is triggered for null expressions
// within CompoundLiteralExpr.
typedef union {
  struct dispatch_object_s *_do;
  struct dispatch_source_s *_ds;
} dispatch_object_t __attribute__((__transparent_union__));
typedef struct dispatch_source_s *dispatch_source_t;

extern __attribute__((visibility("default"))) __attribute__((__nonnull__)) __attribute__((__nothrow__))
void
dispatch_resume(dispatch_object_t object);

@interface AppDelegate : NSObject {
@protected
	dispatch_source_t p;
}
@end
@implementation AppDelegate
- (void)updateDeleteTimer {
	if (p != ((void*)0))
		;
}
- (void)createAndStartDeleteTimer {
  [self updateDeleteTimer];
  dispatch_resume(p); // no warning
}
@end

// Test nil receiver suppression.
// We only suppress on nil receiver if the nil value is directly causing the bug.
@interface Foo {
@public
  int x;
}
- (Foo *)getFooPtr;
@end

Foo *retNil(void) {
  return 0;
}

Foo *retInputOrNil(Foo *p) {
  if (p)
    return p;
  return 0;
}

void idc(Foo *p) {
  if (p)
    ;
}

int testNilReceiver(Foo* fPtr) {
  if (fPtr)
    ;
  // On a path where fPtr is nil, mem should be nil.
  Foo *mem = [fPtr getFooPtr];
  return mem->x; // expected-warning {{Access to instance variable 'x' results in a dereference of a null pointer}}
}

int suppressNilReceiverRetNullCond(Foo* fPtr) {
  unsigned zero = 0;
  fPtr = retInputOrNil(fPtr);
  // On a path where fPtr is nzil, mem should be nil.
  Foo *mem = [fPtr getFooPtr];
  return mem->x;
}

int suppressNilReceiverRetNullCondCast(id fPtr) {
  unsigned zero = 0;
  fPtr = retInputOrNil(fPtr);
  // On a path where fPtr is nzil, mem should be nil.
  Foo *mem = ((id)([(Foo*)(fPtr) getFooPtr]));
  return mem->x;
}

int dontSuppressNilReceiverRetNullCond(Foo* fPtr) {
  unsigned zero = 0;
  fPtr = retInputOrNil(fPtr);
  // On a path where fPtr is nil, mem should be nil.
  // The warning is not suppressed because the receiver being nil is not
  // directly related to the value that triggers the warning.
  Foo *mem = [fPtr getFooPtr];
  if (!mem)
    return 5/zero; // expected-warning {{Division by zero}}
  return 0;
}

int dontSuppressNilReceiverRetNull(Foo* fPtr) {
  unsigned zero = 0;
  fPtr = retNil();
  // On a path where fPtr is nil, mem should be nil.
  // The warning is not suppressed because the receiver being nil is not
  // directly related to the value that triggers the warning.
  Foo *mem = [fPtr getFooPtr];
  if (!mem)
    return 5/zero; // expected-warning {{Division by zero}}
  return 0;
}

int dontSuppressNilReceiverIDC(Foo* fPtr) {
  unsigned zero = 0;
  idc(fPtr);
  // On a path where fPtr is nil, mem should be nil.
  // The warning is not suppressed because the receiver being nil is not
  // directly related to the value that triggers the warning.
  Foo *mem = [fPtr getFooPtr];
  if (!mem)
    return 5/zero; // expected-warning {{Division by zero}}
  return 0;
}
