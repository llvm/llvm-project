// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I {
  int Y;
}
@property int X;
@property int Y;
@property int Z;
@end

@implementation I
@dynamic X; // expected-note {{previous declaration is here}}
@dynamic X; // expected-error {{property 'X' is already implemented}}
@synthesize Y; // expected-note {{previous use is here}}
@synthesize Z=Y; // expected-error {{synthesized properties 'Z' and 'Y' both claim instance variable 'Y'}}
@end

@interface IDEPathCell 
{
@private
    id _gradientStyle;
}

@property (readwrite, assign, nonatomic) id gradientStyle;
@end

@implementation IDEPathCell

@synthesize gradientStyle = _gradientStyle;
- (void)setGradientStyle:(id)value { }

+ (id)_componentCellWithRepresentedObject {
    return self.gradientStyle;
}
@end

@interface rdar11054153
@property int P; // expected-error {{type of property 'P' ('int') does not match type of accessor 'P' ('void')}}
- (void)P; // expected-note {{declared here}}

@property int P1; // expected-warning {{type of property 'P1' does not match type of accessor 'P1'}} 
- (double) P1; // expected-note {{declared here}}

@property int P2; // expected-error {{type of property 'P2' ('int') does not match type of accessor 'P2' ('double *')}}
- (double*)P2; // expected-note {{declared here}}

@end
