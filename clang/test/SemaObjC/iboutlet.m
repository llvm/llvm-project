// RUN: %clang_cc1 -fsyntax-only -fobjc-default-synthesize-properties -Wno-objc-root-class -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -fobjc-default-synthesize-properties -Wno-objc-root-class -verify %s
// rdar://11448209

#define READONLY readonly

@class NSView;

#define IBOutlet __attribute__((iboutlet))

@interface I
@property (getter = MyGetter, readonly, assign) IBOutlet NSView *myView; // expected-note {{property declared here}} \
							// expected-note {{readonly IBOutlet property should be changed to be readwrite}}

@property (readonly) IBOutlet NSView *myView1; // expected-note {{readonly IBOutlet property should be changed to be readwrite}} \
                                               // expected-note {{property declared here}}

@property (getter = MyGetter, READONLY) IBOutlet NSView *myView2;  // expected-note {{property declared here}}

@end

@implementation I // expected-warning 3 {{readonly IBOutlet property when auto-synthesized may not work correctly with 'nib' loader}}
@end
