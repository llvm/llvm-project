// RUN: %clang_cc1 %s -fopenacc -fblocks -verify

// Make sure that Objective-C (and block) types used as OpenACC 'var's are
// diagnosed rather than crashing in recipe generation.

@protocol P @end
@interface Foo @end

// expected-note@+1{{used as field 'i' of composite 'HasId'}}
struct HasId { id i; };
// expected-note@+1{{used as field 'f' of composite 'HasFoo'}}
struct HasFoo { Foo *f; int x; };
// expected-note@+1{{used as field 'b' of composite 'HasBlock'}}
struct HasBlock { void (^b)(void); };

void Reductions(id i, Foo *f, Class c, id<P> ip, id *ipp, HasId hi,
                HasFoo hf, HasBlock hb, void (^blk)(void)) {
  // expected-error@+2{{invalid type 'id' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: i)
  ;
  // expected-error@+2{{invalid type 'Foo *' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(min: f)
  ;
  // expected-error@+2{{invalid type 'Class' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: c)
  ;
  // expected-error@+2{{invalid type 'id<P>' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: ip)
  ;
  // expected-error@+2{{invalid type 'id *' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: ipp)
  ;
  // expected-error@+2{{invalid type 'id' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: hi)
  ;
  // expected-error@+2{{invalid type 'Foo *' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(+: hf)
  ;
  // expected-error@+2{{invalid type 'void (^)()' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(+: hb)
  ;
  // expected-error@+2{{invalid type 'void (^)()' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: blk)
  ;
}

template <typename T>
void TemplReduction(T t) {
  // expected-error@+2{{invalid type 'id' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: t)
  ;
}

void Inst(id i) {
  // expected-note@+1{{in instantiation of function template specialization 'TemplReduction<id>' requested here}}
  TemplReduction(i);
}

// Sub-arrays of ObjC object pointers make no sense, as the size of the pointee
// is not known.
void SubArrays(id i, Foo *f, id *ip) {
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel copy(i[0:1])
  ;
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel copy(f[0:1])
  ;
  // Pointer to an ObjC pointer is a plain C pointer, so this is fine.
#pragma acc parallel copy(ip[0:1])
  ;
}

// Everything else is just a pointer-sized value and is OK to copy around.
void Others(id i, Foo *f, void (^blk)(void)) {
#pragma acc parallel copy(i) copyin(f) copyout(blk) private(i) firstprivate(f)
  ;
}
