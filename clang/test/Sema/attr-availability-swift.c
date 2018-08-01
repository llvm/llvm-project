// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -ast-dump %s | FileCheck %s
//

#if !__has_feature(attribute_availability_with_message)
# error "Missing __has_feature"
#endif

#if __has_feature(attribute_availability_swift)
# warning "okay"
// expected-warning@-1{{okay}}
#else
# error "Missing __has_feature"
#endif

extern int noSwiftGlobal1 __attribute__((availability(swift, unavailable)));
// CHECK: AvailabilityAttr {{.*}}swift 0 0 0 Unavailable "" ""
extern int noSwiftGlobal1 __attribute__((availability(macosx, introduced=10.1))); // okay
// CHECK: AvailabilityAttr {{.*}}macos 10.1 0 0 "" ""
// CHECK: AvailabilityAttr {{.*}}Inherited swift 0 0 0 Unavailable "" ""
extern int noSwiftGlobal1 __attribute__((availability(swift, unavailable, message="and this one has a message"))); // okay
// CHECK: AvailabilityAttr {{.*}}swift 0 0 0 Unavailable "and this one has a message" ""
// CHECK: AvailabilityAttr {{.*}}Inherited macos 10.1 0 0 "" ""
extern int noSwiftGlobal2 __attribute__((availability(swift, introduced=5))); // expected-warning{{only 'unavailable' and 'deprecated' are supported for Swift availability}}
// CHECK: VarDecl
// CHECK-NOT: AvailabilityAttr
extern int noSwiftGlobal3 __attribute__((availability(swift, deprecated, message="t")));
// CHECK: VarDecl
// CHECK: AvailabilityAttr {{.*}}swift 0 1 0 "t" ""
