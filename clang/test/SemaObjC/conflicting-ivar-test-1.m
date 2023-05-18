// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.5 -fsyntax-only -verify -Wno-objc-root-class %s

@interface INTF 
{
@public
	int IVAR; // expected-note {{previous definition is here}}
}
@end

@implementation INTF
{
@private

        int XIVAR; // expected-error {{conflicting instance variable names: 'XIVAR' vs 'IVAR'}}
}
@end



@interface INTF1 
{
@public
	int IVAR;
	int IVAR1; // expected-error {{inconsistent number of instance variables specified}}
}
@end

@implementation INTF1
{
@private

        int IVAR;
}
@end


@interface INTF2 
{
@public
	int IVAR;
}
@end

@implementation INTF2
{
@private

        int IVAR;
	int IVAR1; // expected-error {{inconsistent number of instance variables specified}}
}
@end


@interface INTF3
{
@public
	int IVAR; // expected-note {{previous definition is here}}
}
@end

@implementation INTF3
{
@private

        short IVAR; // expected-error {{instance variable 'IVAR' has conflicting type: 'short' vs 'int'}}
}
@end

@implementation  INTF4 // expected-warning {{cannot find interface declaration for 'INTF4'}}
{
@private

        short IVAR;
}
@end

@interface INTF5
{
  char * ch;
}
@end

@implementation  INTF5
{
}
@end
