@protocol Proto

#ifndef NO_REQUIRED
@required
-(void)ofCourseItisRequired:(int)x;
#endif

#ifndef NO_NOTHING
- (void)nothingSpecified:(int)y;
#endif

#ifndef NO_OPTIONAL
@optional;
- (void)justOptional;
#endif

@end

@protocol Proto2

#ifndef NO_NOTHING
// Effectively a @required.
- (void)nothingSpecified2:(int)y;
#endif

#ifndef NO_REQUIRED
@required
-(void)ofCourseItisRequired2:(int)x;
#endif

#ifndef NO_OPTIONAL
@optional;
- (void)justOptional2;
#endif

@end

@interface Base
@end

// Initiate in the @interface when the interface has missing @required
// declarations.
@interface I1 : Base<Proto>
// CHECK1: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
#ifdef DEF_REQUIRED
-(void)ofCourseItisRequired:(int)x;
#endif
#ifdef DEF_NOTHING
- (void)nothingSpecified:(int)y;
#endif
#ifdef DEF_OPTIONAL
- (void)justOptional;
#endif
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DNO_REQUIRED -DNO_OPTIONAL | FileCheck --check-prefix=CHECK1 %s
// CHECK-NO: Failed to initiate the refactoring action (All of the @required methods are there)!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DNO_REQUIRED -DNO_NOTHING -DNO_OPTIONAL 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DNO_REQUIRED -DNO_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DDEF_REQUIRED | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DDEF_NOTHING | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DDEF_OPTIONAL | FileCheck --check-prefix=CHECK1 %s

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:43:1 %s -DDEF_REQUIRED -DDEF_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

@interface I1(Category) <Proto2>
// CHECK2: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
#ifdef DEF_REQUIRED
-(void)ofCourseItisRequired2:(int)x;
#endif
#ifdef DEF_NOTHING
- (void)nothingSpecified2:(int)y;
#endif
#ifdef DEF_OPTIONAL
- (void)justOptional2;
#endif
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DNO_REQUIRED -DNO_OPTIONAL | FileCheck --check-prefix=CHECK2 %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DNO_REQUIRED -DNO_NOTHING -DNO_OPTIONAL 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DNO_REQUIRED -DNO_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DDEF_REQUIRED | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DDEF_NOTHING | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DDEF_OPTIONAL | FileCheck --check-prefix=CHECK2 %s

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:67:1 %s -DDEF_REQUIRED -DDEF_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// Initiate in the @implementatino when the implementation has missing @required
// methods.
@implementation I1
// CHECK3: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
#ifdef IMPL_REQUIRED
-(void)ofCourseItisRequired:(int)x { }
#endif
#ifdef IMPL_NOTHING
- (void)nothingSpecified:(int)y { }
#endif
#ifdef IMPL_OPTIONAL
- (void)justOptional { }
#endif
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DNO_REQUIRED -DNO_OPTIONAL | FileCheck --check-prefix=CHECK3 %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DNO_REQUIRED -DNO_NOTHING -DNO_OPTIONAL 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DNO_REQUIRED -DNO_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_REQUIRED | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_NOTHING | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_OPTIONAL | FileCheck --check-prefix=CHECK3 %s

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_REQUIRED -DIMPL_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DDEF_REQUIRED -DDEF_NOTHING -DDEF_OPTIONAL | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_REQUIRED -DDEF_REQUIRED -DDEF_NOTHING | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_NOTHING -DDEF_REQUIRED -DDEF_NOTHING | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:92:1 %s -DIMPL_OPTIONAL -DDEF_REQUIRED -DDEF_NOTHING -DDEF_OPTIONAL | FileCheck --check-prefix=CHECK3 %s

@implementation I1 (Category)
// CHECK4: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
#ifdef IMPL_REQUIRED
-(void)ofCourseItisRequired2:(int)x { }
#endif
#ifdef IMPL_NOTHING
- (void)nothingSpecified2:(int)y { }
#endif
#ifdef IMPL_OPTIONAL
- (void)justOptional2 { }
#endif
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DNO_REQUIRED -DNO_OPTIONAL | FileCheck --check-prefix=CHECK4 %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DNO_REQUIRED -DNO_NOTHING -DNO_OPTIONAL 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DNO_REQUIRED -DNO_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_REQUIRED | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_NOTHING | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_OPTIONAL | FileCheck --check-prefix=CHECK4 %s

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_REQUIRED -DIMPL_NOTHING 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DDEF_REQUIRED -DDEF_NOTHING -DDEF_OPTIONAL | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_REQUIRED -DDEF_REQUIRED -DDEF_NOTHING | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_NOTHING -DDEF_REQUIRED -DDEF_NOTHING | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:120:1 %s -DIMPL_OPTIONAL -DDEF_REQUIRED -DDEF_NOTHING -DDEF_OPTIONAL | FileCheck --check-prefix=CHECK4 %s
