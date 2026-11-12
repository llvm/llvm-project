// RUN: not %clang_cc1 %s -fdiagnostics-print-source-range-info 2>&1 | FileCheck %s --strict-whitespace

template<typename T> class C {};
template<int> class D {};

void g() {
  // The range ending in the first > character does not extend to the second >
  // character.
  // CHECK:      :{[[@LINE+3]]:5-[[@LINE+3]]:11}: error:
  // CHECK-NEXT:   D<C<int>> a;
  // CHECK-NEXT:     ^~~~~~{{$}}
  D<C<int>> a;

  // The range ending in the second > character does not extend to the third >
  // character.
  // CHECK:      :{[[@LINE+3]]:5-[[@LINE+3]]:14}: error:
  // CHECK-NEXT:   D<C<C<int>>> b;
  // CHECK-NEXT:     ^~~~~~~~~{{$}}
  D<C<C<int>>> b;
}

template<int> int V;
// Here, we split the >>= token into a > followed by a >=.
// Then we split the >= token into a > followed by an =,
// which we merge with the other = to form an ==.
// CHECK:      error: a space is required
// CHECK-NEXT: int k = V<C<int>>==0;
// CHECK-NEXT:                ^~{{$}}
// CHECK-NEXT:                > >{{$}}
// CHECK:      error: a space is required
// CHECK-NEXT: int k = V<C<int>>==0;
// CHECK-NEXT:                 ^~{{$}}
// CHECK-NEXT:                 > ={{$}}
// CHECK:      :{[[@LINE+3]]:11-[[@LINE+3]]:17}: error:
// CHECK-NEXT: int k = V<C<int>>==0;
// CHECK-NEXT:           ^~~~~~{{$}}
int k = V<C<int>>==0;

template<typename> int W;
// CHECK:      :{[[@LINE+3]]:9-[[@LINE+3]]:18}{[[@LINE+3]]:20-[[@LINE+3]]:22}: error: comparison
// CHECK-NEXT: int l = W<C<int>>==&k;
// CHECK-NEXT:         ~~~~~~~~~^ ~~{{$}}
int l = W<C<int>>==&k;
