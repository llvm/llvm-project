// RUN: clang-tidy %s -checks=-*,performance-unnecessary-value-param -- \
// RUN:   -xobjective-c++ -fobjc-abi-version=2 -fobjc-arc | count 0

#if !__has_feature(objc_arc)
#error Objective-C ARC not enabled as expected
#endif

// Passing an Objective-C ARC-managed object to a C function should
// not raise performance-unnecessary-value-param.
void foo(id object) { }

// Same for explcitly non-ARC-managed Objective-C objects.
void bar(__unsafe_unretained id object) { }

// Same for Objective-c classes.
void baz(Class c) { }
