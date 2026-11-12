// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface testClass
@end

@class NSArray;

@implementation testClass

static NSArray* prefixArray[] = @"BEGIN:", @"END:", @"VERSION:", @"N:", @"FN:", @"TEL;", @"TEL:", nil; // expected-error {{array initializer must be an initializer list}} \
												       // expected-error {{expected identifier or '('}} \
												       // expected-error {{expected ';' after top level declarator}}

static NSString* prefixArray1[] = @"BEGIN:", @"END:", @"VERSION:", @"N:", @"FN:", @"TEL;", @"TEL:", nil; // expected-error {{unknown type name 'NSString'}} \
													 // expected-error {{expected identifier or '('}} \
													 // expected-error {{expected ';' after top level declarator}}

static char* cArray[] = "BEGIN:", "END";	// expected-error {{array initializer must be an initializer list}} \
						// expected-error {{expected identifier or '('}} \
						// expected-error {{expected ';' after top level declarator}}

@end
