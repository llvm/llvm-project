// RUN: %clang %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clang %target_itanium_abi_host_triple %t.o -o %t.out -framework Foundation
// RUN: %test_debuginfo %s %t.out
//
// REQUIRES: system-darwin
// Radar 8757124

// DEBUGGER: break 25
// DEBUGGER: r
// DEBUGGER: po thing
// CHECK: aaa

#import <Foundation/Foundation.h>

int main (int argc, const char * argv[]) {

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSArray *things = [NSArray arrayWithObjects:@"one", @"two", @"three" , nil];
    for (NSString *thing in things) {
        NSLog (@"%@", thing);
    }

    things = [NSArray arrayWithObjects:@"aaa", @"bbb", @"ccc" , nil];
    for (NSString *thing in things) {
        NSLog (@"%@", thing);
    }
    [pool release];
    return 0;
}


