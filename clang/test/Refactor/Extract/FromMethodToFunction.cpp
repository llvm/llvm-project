// RUN: clang-refactor extract -selection=test:%s %s -- -std=c++11 2>&1 | grep -v CHECK | FileCheck --check-prefixes=CHECK,CHECK-INNER %s
// RUN: clang-refactor extract -selection=test:%s %s -- -std=c++11 -DMULTIPLE 2>&1 | grep -v CHECK | FileCheck --check-prefixes=CHECK,CHECK-OUTER %s

#ifdef MULTIPLE
class OuterClass {
#define PREFIX OuterClass ::
#else
#define PREFIX
#endif

class AClass {

  int method(int x) {
    return /*range inner=->+0:38*/1 + 2 * 2;
  }
// CHECK-INNER: 1 'inner' results:
// CHECK-INNER:      static int extracted() {
// CHECK-INNER-NEXT: return 1 + 2 * 2;{{$}}
// CHECK-INNER-NEXT: }{{[[:space:]].*}}
// CHECK-INNER-NEXT: class AClass {

// CHECK-OUTER: 1 'inner' results:
// CHECK-OUTER:      static int extracted() {
// CHECK-OUTER-NEXT: return 1 + 2 * 2;{{$}}
// CHECK-OUTER-NEXT: }{{[[:space:]].*}}
// CHECK-OUTER-NEXT: class OuterClass {

  int otherMethod(int x);
};

#ifdef MULTIPLE
};
#endif

int PREFIX AClass::otherMethod(int x) {
  return /*range outofline=->+0:46*/2 * 2 - 1;
}
// CHECK: 1 'outofline' results:
// CHECK:      static int extracted() {
// CHECK-NEXT: return 2 * 2 - 1;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: int PREFIX AClass::otherMethod
