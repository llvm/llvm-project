// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -Wno-objc-root-class -Warc-repeated-use-of-weak -fobjc-runtime-has-weak -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -fobjc-arc -Wno-objc-root-class -Warc-repeated-use-of-weak -fobjc-runtime-has-weak -verify %s
// rdar://11448209

#define READONLY readonly

@class NSView;

IB_DESIGNABLE @interface I
@property (getter = MyGetter, readonly, assign) IBOutlet NSView *myView; // expected-warning {{readonly IBOutlet property 'myView' when auto-synthesized may not work correctly with 'nib' loader}} expected-note {{property should be changed to be readwrite}}

IBInspectable @property (readonly) IBOutlet NSView *myView1; // expected-warning {{readonly IBOutlet property 'myView1' when auto-synthesized may not work correctly with 'nib' loader}} expected-note {{property should be changed to be readwrite}}

@property (getter = MyGetter, READONLY) IBOutlet NSView *myView2; // expected-warning {{readonly IBOutlet property 'myView2' when auto-synthesized may not work correctly with 'nib' loader}}

@end

@implementation I
@end


// rdar://13123861
@class UILabel;

@interface NSObject @end

@interface RKTFHView : NSObject
@property( readonly ) __attribute__((iboutlet)) UILabel *autoReadOnlyReadOnly; // expected-warning {{readonly IBOutlet property 'autoReadOnlyReadOnly' when auto-synthesized may not work correctly with 'nib' loader}} expected-note {{property should be changed to be readwrite}}
@property( readonly ) __attribute__((iboutlet)) UILabel *autoReadOnlyReadWrite;
@property( readonly ) __attribute__((iboutlet)) UILabel *synthReadOnlyReadWrite;
@end

@interface RKTFHView()
@property( readwrite ) __attribute__((iboutlet)) UILabel *autoReadOnlyReadWrite;
@property( readwrite ) __attribute__((iboutlet)) UILabel *synthReadOnlyReadWrite;
@end

@implementation RKTFHView
@synthesize synthReadOnlyReadWrite=_synthReadOnlyReadWrite;
@end

// rdar://15885642
@interface WeakOutlet 
@property int Number;
@property IBOutlet __weak WeakOutlet* WeakProp;
@end

WeakOutlet* func() {
  __weak WeakOutlet* pwi;
  pwi.WeakProp = (WeakOutlet*)0;
  pwi.WeakProp = pwi.WeakProp;
  return pwi.WeakProp;
}

WeakOutlet* func2(WeakOutlet* pwi) {
  [[pwi WeakProp] setNumber:0];
  [[pwi WeakProp] setNumber:1];
  return [pwi WeakProp];
}
