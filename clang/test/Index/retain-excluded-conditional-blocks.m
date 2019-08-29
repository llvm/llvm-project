// RUN: c-index-test -retain-excluded-conditional-blocks %s | FileCheck %s

#include <stdint.h>

// CHECK: TypedefDecl=intptr_t

// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=MyCls
@interface MyCls
// CHECK: [[@LINE+1]]:8: ObjCInstanceMethodDecl=some_meth
-(void)some_meth;
@end

#if 1
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test1
@interface Test1 @end
#else
// CHECK: [[@LINE+1]]:12:
@interface Test2 @end
#endif

#if 0
// CHECK: [[@LINE+1]]:12:
@interface Test3 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test4
@interface Test4 @end
#endif

#if SOMETHING_NOT_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test5
@interface Test5 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test6
@interface Test6 @end
#endif

#define SOMETHING_DEFINED 1
#if SOMETHING_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test7
@interface Test7 @end
#else
// CHECK: [[@LINE+1]]:12:
@interface Test8 @end
#endif

#if defined(SOMETHING_NOT_DEFINED)
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test9
@interface Test9 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test10
@interface Test10 @end
#endif

#if defined(SOMETHING_DEFINED)
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test11
@interface Test11 @end
#else
// CHECK: [[@LINE+1]]:12:
@interface Test12 @end
#endif

#if SOMETHING_NOT_DEFINED1
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test13
@interface Test13 @end
#elif SOMETHING_NOT_DEFINED2
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test14
@interface Test14 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test15
@interface Test15 @end
#endif

#ifdef SOMETHING_NOT_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test19
@interface Test19 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test20
@interface Test20 @end
#endif

#ifdef SOMETHING_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test21
@interface Test21 @end
#else
// CHECK: [[@LINE+1]]:12:
@interface Test22 @end
#endif

#ifndef SOMETHING_NOT_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test23
@interface Test23 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test24
@interface Test24 @end
#endif

#ifndef SOMETHING_DEFINED
// CHECK: [[@LINE+1]]:12:
@interface Test25 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test26
@interface Test26 @end
#endif

#if 1 < SOMETHING_NOT_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test27
@interface Test27 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test28
@interface Test28 @end
#endif

#if SOMETHING_NOT_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test29
@interface Test29 @end
#endif

#ifdef SOMETHING_NOT_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test30
@interface Test30 @end
#endif

#ifdef SOMETHING_DEFINED
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test31
@interface Test31 @end
#elif !defined(SOMETHING_NOT_DEFINED)
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test32
@interface Test32 @end
#else
// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=Test33
@interface Test33 @end
#endif
