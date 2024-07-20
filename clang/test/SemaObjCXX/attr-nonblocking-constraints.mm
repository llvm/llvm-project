// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify %s

#pragma clang diagnostic ignored "-Wperf-constraint-implies-noexcept"

// Objective-C
@interface OCClass
- (void)method;
@end

void nl14(OCClass *oc) [[clang::nonblocking]] {
	[oc method]; // expected-warning {{'nonblocking' function must not access an ObjC method or property}}
}
void nl15(OCClass *oc) {
	[oc method]; // expected-note {{function cannot be inferred 'nonblocking' because it accesses an ObjC method or property}}
}
void nl16(OCClass *oc) [[clang::nonblocking]] {
	nl15(oc); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function 'nl15'}}
}

