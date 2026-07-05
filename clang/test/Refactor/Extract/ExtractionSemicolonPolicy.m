// RUN: clang-refactor extract -selection=test:%s %s -- 2>&1 | grep -v CHECK | FileCheck %s

@interface NSArray
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

void extractStatementNoSemiObjCFor(NSArray *array) {
  /*range astmt=->+2:4*/for (id i in array) {
    int x = 0;
  }
}
// CHECK: 1 'astmt' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: for (id i in array) {
// CHECK-NEXT: int x = 0;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}

void extractStatementNoSemiSync(void) {
  id lock;
  /*range bstmt=->+2:4*/@synchronized(lock) {
    int x = 0;
  }
}
// CHECK: 1 'bstmt' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: @synchronized(lock) {
// CHECK-NEXT: int x = 0;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}

void extractStatementNoSemiAutorel(void) {
  /*range cstmt=->+2:4*/@autoreleasepool {
    int x = 0;
  }
}
// CHECK: 1 'cstmt' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: @autoreleasepool {
// CHECK-NEXT: int x = 0;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}

void extractStatementNoSemiTryFinalllllly(void) {
  /*range dstmt=->+3:4*/@try {
    int x = 0;
  } @finally {
  }
}
// CHECK: 1 'dstmt' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: @try {
// CHECK-NEXT: int x = 0;
// CHECK-NEXT: } @finally {
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
