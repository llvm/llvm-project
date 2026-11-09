// RUN: %clang_cc1 %s -fopenacc -verify

// Sub-arrays require a C/C++ pointer or array base. ObjC object pointers are
// rejected, as the size of the pointee isn't known.

@interface Foo @end
@protocol P @end

void f(id i, Foo *o, id<P> ip, Class c, id *pp, id arr[2], Foo *foos[4]) {
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel copy(i[0:1])
  ;
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel copyin(o[0:1])
  ;
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel present(ip[1:2])
  ;
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel create(c[0:1])
  ;
  // Pointers to / arrays of ObjC pointers are plain C pointers/arrays.
#pragma acc parallel copy(pp[0:1], arr[0:2], foos[1:2])
  ;
}
