// RUN: %clang_cc1 %s -fopenacc -ffixed-point -verify

// Fixed-point types are scalar, but we cannot generate init recipes for them,
// so they must be diagnosed rather than crashing.

struct HasFract {
  // expected-note@+1{{used as field 'f' of composite 'HasFract'}}
  _Fract f;
};

void Reductions(_Fract f, _Accum a, _Sat _Fract sf, _Fract arr[2],
                struct HasFract hf) {
  // expected-error@+2{{invalid type '_Fract' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: f)
  ;
  // expected-error@+2{{invalid type '_Accum' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(min: a)
  ;
  // expected-error@+2{{invalid type '_Sat _Fract' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(+: sf)
  ;
  // expected-error@+2{{invalid type '_Fract' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: arr[0:2])
  ;
  // expected-error@+2{{invalid type '_Fract' used in OpenACC 'reduction' variable reference}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable}}
#pragma acc parallel reduction(max: hf)
  ;
}

// Other clauses just copy the value around, so they are fine.
void Others(_Fract f) {
#pragma acc parallel copy(f) private(f) firstprivate(f)
  ;
}
