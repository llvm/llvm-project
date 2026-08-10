// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end

@interface NSObject <NSObject> {} @end

@interface _NSServicesInContextMenu : NSObject {
    id _requestor;
    NSObject *_appleEventDescriptor;
}

@property (retain, nonatomic) id requestor;
@property (retain, nonatomic) id appleEventDescriptor;

@end

@implementation _NSServicesInContextMenu

@synthesize requestor = _requestor, appleEventDescriptor = _appleEventDescriptor;

@end

@class NSString;

@protocol MyProtocol
- (NSString *)stringValue;
@end

@interface MyClass : NSObject {
  id  _myIvar;
}
@property (readwrite, retain) id<MyProtocol> myIvar;
@end

@implementation MyClass
@synthesize myIvar = _myIvar;
@end


@interface BadPropClass
{
 int _awesome;
}

@property (readonly) int; // expected-warning {{declaration does not declare anything}}
@property (readonly) ; // expected-error {{type name requires a specifier or qualifier}}
@property (readonly) int : 4; // expected-error {{property requires fields to be named}}


// test parser recovery
@property (                           // expected-note {{to match this '('}}
           readonly getter=isAwesome) // expected-error {{expected ')'}}
           
  int _awesome;
@property (readonlyx) // expected-error {{unknown property attribute 'readonlyx'}}
  int _awesome2;

@property (    // expected-note {{to match this '('}}
           +)  // expected-error {{expected ')'}}
           
  int _awesome3;

@end

@protocol PVImageViewProtocol
@property int inEyeDropperMode;
@end

@interface Cls
@property int inEyeDropperMode;
@end

@interface PVAdjustColor @end

@implementation PVAdjustColor

- xx {
  id <PVImageViewProtocol> view;
  Cls *c;

  c.inEyeDropperMode = 1;
  view.inEyeDropperMode = 1;
}
@end

@interface MyStyleIntf 
{
    int _myStyle;
}

@property(readonly) int myStyle;

- (float)setMyStyle:(int)style;
@end

@class MDAInstance; // expected-note {{forward declaration of class here}}

@interface MDATestDocument
@property(retain) MDAInstance *instance;
@end

id f0(MDATestDocument *d) {
  return d.instance.path; // expected-error {{property 'path' cannot be found in forward class object 'MDAInstance'}}
}

@interface UIView @end

@interface FRFakeBannerView : UIView
@end

@interface FRAdCollectionViewCell
@property (nonatomic, weak, readonly) UIView *bannerView;
@end

@interface FRAdCollectionViewCell () 

@property (nonatomic, weak, readwrite) FRFakeBannerView *bannerView;

@end
