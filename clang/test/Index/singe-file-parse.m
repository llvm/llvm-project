// RUN: c-index-test -single-file-parse %s | FileCheck %s

#include <stdint.h>

// CHECK-NOT: TypedefDecl=intptr_t

// CHECK: [[@LINE+1]]:12: ObjCInterfaceDecl=MyCls
@interface MyCls
// CHECK: [[@LINE+1]]:8: ObjCInstanceMethodDecl=some_meth
-(void)some_meth;
@end
