// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class -Wmissing-selector-name %s
// rdar://12263549

@interface Super @end
@interface INTF : Super
-(void) Name1:(id)Arg1 Name2:(id)Arg2; // Name1:Name2:
-(void) Name1:(id) Name2:(id)Arg2; // expected-warning {{'Name2' used as the name of the previous parameter rather than as part of the selector}} \
				   // expected-note {{introduce a parameter name to make 'Name2' part of the selector}} \
				   // expected-note {{or insert whitespace before ':' to use 'Name2' as parameter name and have an empty entry in the selector}}
-(void) Name1:(id)Arg1 Name2:(id)Arg2 Name3:(id)Arg3; // Name1:Name2:Name3:
-(void) Name1:(id)Arg1 Name2:(id) Name3:(id)Arg3; // expected-warning {{'Name3' used as the name of the previous parameter rather than as part of the selector}} \
				   // expected-note {{introduce a parameter name to make 'Name3' part of the selector}} \
				   // expected-note {{or insert whitespace before ':' to use 'Name3' as parameter name and have an empty entry in the selector}}
- method:(id) second:(id)second; // expected-warning {{'second' used as the name of the previous parameter rather than as part of the selector}} \
				   // expected-note {{introduce a parameter name to make 'second' part of the selector}} \
				   // expected-note {{or insert whitespace before ':' to use 'second' as parameter name and have an empty entry in the selector}} \
				   // expected-note {{method definition for 'method::' not found}}
                                 
@end

@implementation INTF // expected-warning {{incomplete implementation}}
-(void) Name1:(id)Arg1 Name2:(id)Arg2{}
-(void) Name1:(id) Name2:(id)Arg2 {} // expected-warning {{'Name2' used as the name of the previous parameter rather than as part of the selector}} \
					// expected-note {{introduce a parameter name to make 'Name2' part of the selector}} \
 					// expected-note {{or insert whitespace before ':' to use 'Name2' as parameter name and have an empty entry in the selector}}
-(void) Name1:(id)Arg1 Name2:(id)Arg2 Name3:(id)Arg3 {}
-(void) Name1:(id)Arg1 Name2:(id) Name3:(id)Arg3 {} // expected-warning {{'Name3' used as the name of the previous parameter rather than as part of the selector}} \
					// expected-note {{introduce a parameter name to make 'Name3' part of the selector}} \
 					// expected-note {{or insert whitespace before ':' to use 'Name3' as parameter name and have an empty entry in the selector}}
- method:(id)first second:(id)second {return 0; }
@end
