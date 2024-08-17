// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -fobjc-exceptions -verify %s

#pragma clang diagnostic ignored "-Wperf-constraint-implies-noexcept"

// Objective-C
@interface OCClass
- (void)method;
@end

void nb1(OCClass *oc) [[clang::nonblocking]] {
	[oc method]; // expected-warning {{'nonblocking' function must not access ObjC methods or properties}}
}
void nb2(OCClass *oc) {
	[oc method]; // expected-note {{function cannot be inferred 'nonblocking' because it accesses an ObjC method or property}}
}
void nb3(OCClass *oc) [[clang::nonblocking]] {
	nb2(oc); // expected-warning {{'nonblocking' function must not call non-'nonblocking' function 'nb2'}}
}

void nb4() [[clang::nonblocking]] {
	@try {
		@throw @"foo"; // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	}
	@catch (...) { // expected-warning {{'nonblocking' function must not throw or catch exceptions}}
	}
}
