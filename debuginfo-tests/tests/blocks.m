// RUN: %clang %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clang %target_itanium_abi_host_triple %t.o -o %t.out -framework Foundation
// RUN: %test_debuginfo %s %t.out

// REQUIRES: system-darwin
// Radar 9279956

// DEBUGGER: break 31
// DEBUGGER: r
// DEBUGGER: p m2
// CHECK: ${{[0-9]}} = 1
// DEBUGGER: p dbTransaction
// CHECK: ${{[0-9]}} = 0
// DEBUGGER: p master
// CHECK: ${{[0-9]}} = 0

#include <Cocoa/Cocoa.h>

extern void foo(void(^)(void));

@interface A:NSObject @end
@implementation A
- (void) helper {
 int master = 0;
 __block int m2 = 0;
 __block int dbTransaction = 0;
 int (^x)(void) = ^(void) { (void) self;
	(void) master;
	(void) dbTransaction;
	m2++;
	return m2;
	};
  master = x();
}
@end

void foo(void(^x)(void)) {}

int main() {
	A *a = [A alloc];
	[a helper];
	return 0;
}
