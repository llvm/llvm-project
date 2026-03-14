// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify -Wno-objc-root-class %s

@interface Subtask
{
  id _delegate;
}
@property(nonatomic,readwrite,assign)   id __weak       delegate;  // expected-error {{unsafe_unretained property 'delegate' may not also be declared __weak}}
@end

@implementation Subtask
@synthesize delegate = _delegate;
@end

 
@interface PVSelectionOverlayView2 
{
 id __weak _selectionRect;  // expected-error {{cannot create __weak reference because the current deployment target does not support weak references}} expected-error {{existing instance variable '_selectionRect' for property 'selectionRect' with assign attribute must be __unsafe_unretained}}
}

@property(assign) id selectionRect; // expected-note {{property declared here}}

@end

@implementation PVSelectionOverlayView2

@synthesize selectionRect = _selectionRect; // expected-note {{property synthesized here}}
@end

