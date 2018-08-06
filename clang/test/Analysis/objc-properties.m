// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.osx.cocoa.DirectIvarAssignment -verify -fblocks %s

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
@end

@interface MyClass;
@end
@interface TestProperty :NSObject {
  MyClass *_Z;
  id _nonSynth;
}

  @property (assign, nonatomic) MyClass* A; // explicitly synthesized, not implemented, non-default ivar name

  @property (assign) MyClass* X;  // automatically synthesized, not implemented

  @property (assign, nonatomic) MyClass* Y; // automatically synthesized, implemented

  @property (assign, nonatomic) MyClass* Z; // non-synthesized ivar, implemented setter
  @property (readonly) id nonSynth;  // non-synthesized, explicitly implemented to return ivar with expected name
  
  - (id) initWithPtr:(MyClass*) value;
  - (id) myInitWithPtr:(MyClass*) value;
  - (void) someMethod: (MyClass*)In;
@end

@implementation TestProperty
  @synthesize A = __A;
  
  - (id) initWithPtr: (MyClass*) value {
    _Y = value; // no-warning
    return self;
  }

  - (id) copyWithPtrY: (TestProperty*) value {
    TestProperty *another = [[TestProperty alloc] init];
    another->_Y = value->_Y; // no-warning
    return another;
  }

  - (id) myInitWithPtr: (MyClass*) value {
    _Y = value; // no-warning
    return self;
  }
  
  - (void) setY:(MyClass*) NewValue {
    _Y = NewValue; // no-warning
  }

  - (void) setZ:(MyClass*) NewValue {
    _Z = NewValue; // no-warning
  }

  - (id)nonSynth {
      return _nonSynth;
  }

  - (void) someMethod: (MyClass*)In {
    (__A) = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _X = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _Y = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _Z = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _nonSynth = 0; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
  }
@end