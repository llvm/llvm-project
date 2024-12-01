// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -fobjc-exceptions -verify -Wfunction-effects %s

#pragma clang diagnostic ignored "-Wperf-constraint-implies-noexcept"

// Objective-C
@interface OCClass
- (void)method;
@end

void nb1(OCClass *oc) [[clang::nonblocking]] {
	[oc method]; // expected-warning {{function with 'nonblocking' attribute must not access ObjC methods or properties}}
}
void nb2(OCClass *oc) {
	[oc method]; // expected-note {{function cannot be inferred 'nonblocking' because it accesses an ObjC method or property}}
}
void nb3(OCClass *oc) [[clang::nonblocking]] {
	nb2(oc); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'nb2'}}
}

void nb4() [[clang::nonblocking]] {
	@try {
		@throw @"foo"; // expected-warning {{function with 'nonblocking' attribute must not throw or catch exceptions}}
	}
	@catch (...) { // expected-warning {{function with 'nonblocking' attribute must not throw or catch exceptions}}
	}
	@finally { // expected-warning {{function with 'nonblocking' attribute must not throw or catch exceptions}}
	}
}

@class Lock;
extern Lock *someLock;

void nb5() [[clang::nonblocking]] {
	@autoreleasepool { // expected-warning {{function with 'nonblocking' attribute must not access ObjC methods or properties}}
	}

	@synchronized(someLock) { // expected-warning {{function with 'nonblocking' attribute must not access ObjC methods or properties}}
	}
}
