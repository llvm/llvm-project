// RUN: %clang_cc1 %s -fopenacc -verify

struct CompositeOfScalars {
  int I;
  float F;
  short J;
  char C;
  double D;
};

struct CompositeHasComposite {
  int I;
  float F;
  short J;
  char C;
  double D;
  struct CompositeOfScalars COS; // #COS_FIELD
};
void uses() {

  int I;
  float F;
  int Array[5];
  CompositeOfScalars CoS;
  CompositeHasComposite ChC;

#pragma acc serial
  {
#pragma acc loop reduction(+:CoS, I, F)
    for(int i = 0; i < 5; ++i){}
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:Array)
    for(int i = 0; i < 5; ++i){}
  }

#pragma acc serial
  {
  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc loop reduction(+:ChC)
    for(int i = 0; i < 5; ++i){}
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:I)
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:I)
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:I)
    for(int i = 0; i < 5; ++i) {
#pragma acc serial
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-4{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:I)
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc serial reduction(+:I)
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-2{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:I)
  for(int i = 0; i < 5; ++i){}

#pragma acc serial
#pragma acc loop reduction(&:I)
  for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (+ vs &)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc serial reduction(+:I)
    ;
  }

#pragma acc parallel
  {
#pragma acc loop reduction(+:I) gang(dim:1)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'gang' clause with a 'dim' value greater than 1 cannot appear on the same 'loop' construct as a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:2)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause with a 'dim' value greater than 1}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc loop gang(dim:2) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause with a 'dim' value greater than 1}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc loop gang gang(dim:1) gang(dim:2) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(1, 2)
  {
    // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'gang' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop gang(dim:1) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(2, 3)
  {
    // expected-error@+3{{OpenACC 'gang' clause cannot appear on the same 'loop' construct as a 'reduction' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'reduction' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:1)
    for(int i = 0; i < 5; ++i) {
    }
  }

  CompositeHasComposite CoCArr[5];
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of array type 'CompositeHasComposite'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc loop reduction(+:CoCArr)
    for(int i = 0; i < 5; ++i);
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of array type 'CompositeHasComposite[5]'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc loop reduction(+:CoCArr[3])
    for(int i = 0; i < 5; ++i);
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of sub-array type 'CompositeHasComposite'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc loop reduction(+:CoCArr[1:1])
    for(int i = 0; i < 5; ++i);
}

template<typename IntTy, typename CoSTy, typename ChCTy, unsigned One,
         unsigned Two>
void templ_uses() {
  IntTy I;
  IntTy Array[5];
  CoSTy CoS;
  ChCTy ChC;

#pragma acc serial
  {
#pragma acc loop reduction(+:CoS, I)
    for(int i = 0; i < 5; ++i){}
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:Array)
    for(int i = 0; i < 5; ++i){}
  }

#pragma acc serial
  {
  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc loop reduction(+:ChC)
    for(int i = 0; i < 5; ++i){}
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:I)
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:Array[3])
    for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:Array[3])
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:Array[0:3])
    for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:Array[1:4])
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc serial
  {
#pragma acc loop reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
    // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc serial reduction(&:I)
      for(int i = 0; i < 5; ++i) {
      }
    }
  }

#pragma acc parallel
  {
#pragma acc loop reduction(+:I) gang(dim:One)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'gang' clause with a 'dim' value greater than 1 cannot appear on the same 'loop' construct as a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:2)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause with a 'dim' value greater than 1}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc loop gang(dim:2) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }
#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'gang' clause with a 'dim' value greater than 1 cannot appear on the same 'loop' construct as a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:Two)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel
  {
  // expected-error@+2{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause with a 'dim' value greater than 1}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc loop gang(dim:Two) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }


#pragma acc parallel num_gangs(One)
  {
#pragma acc loop reduction(+:I) gang(dim:One)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(Two, 1)
  {
    // expected-error@+3{{OpenACC 'gang' clause cannot appear on the same 'loop' construct as a 'reduction' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'reduction' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:One)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(Two, 1)
  {
    // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'gang' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop gang(dim:One) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(One)
  {
#pragma acc loop reduction(+:I) gang(dim:1)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(Two, 1)
  {
    // expected-error@+3{{OpenACC 'gang' clause cannot appear on the same 'loop' construct as a 'reduction' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'reduction' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:1)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(Two, 1)
  {
    // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'gang' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop gang(dim:1) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(1)
  {
#pragma acc loop reduction(+:I) gang(dim:One)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(2, 1)
  {
    // expected-error@+3{{OpenACC 'gang' clause cannot appear on the same 'loop' construct as a 'reduction' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'reduction' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop reduction(+:I) gang(dim:One)
    for(int i = 0; i < 5; ++i) {
    }
  }

#pragma acc parallel num_gangs(2, 1)
  {
    // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'loop' construct as a 'gang' clause inside a compute construct with a 'num_gangs' clause with more than one argument}}
    // expected-note@+2{{previous 'gang' clause is here}}
    // expected-note@-4{{previous 'num_gangs' clause is here}}
#pragma acc loop gang(dim:One) reduction(+:I)
    for(int i = 0; i < 5; ++i) {
    }
  }
}

void inst() {
  // expected-note@+1{{in instantiation of function template specialization}}
  templ_uses<int, CompositeOfScalars, CompositeHasComposite, 1, 2>();
}


