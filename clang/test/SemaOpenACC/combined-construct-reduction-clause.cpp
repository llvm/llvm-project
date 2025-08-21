// RUN: %clang_cc1 %s -fopenacc -verify

struct CompositeOfScalars {
  int I;
  float F;
  short J;
  char C;
  double D;
  _Complex float CF;
  _Complex double CD;
};

struct CompositeHasComposite {
  int I;
  float F;
  short J;
  char C;
  double D;
  _Complex float CF;
  _Complex double CD;
  struct CompositeOfScalars COS; // #COS_FIELD
};

  // All of the type checking is done for compute and loop constructs, so only check the basics + the parts that are combined specific.
void uses(unsigned Parm) {
  struct CompositeOfScalars CoS;
  struct CompositeHasComposite ChC;
  int I;
  float F;
  int Array[5];

  // legal on all 3 kinds of combined constructs
#pragma acc parallel loop reduction(+:Parm)
  for(int i = 0; i < 5; ++i);

#pragma acc serial loop reduction(&: CoS, I, F)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop reduction(min: CoS, Array[I], Array[0:I])
  for(int i = 0; i < 5; ++i);

  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel loop reduction(&: ChC)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop reduction(+:Parm) num_gangs(I)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel loop' construct with a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc parallel loop reduction(+:Parm) num_gangs(I, I)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop num_gangs(I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel loop' construct with a 'num_gangs' clause with more than 1 argument}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc parallel loop num_gangs(I, I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);

  // Reduction cannot appear on a loop with a 'gang' of dim>1.
#pragma acc parallel loop gang(dim:1) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause with a 'dim' value greater than 1}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc parallel loop gang(dim:2) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop reduction(+:Parm) gang(dim:1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'gang' clause with a 'dim' value greater than 1 cannot appear on the same 'parallel loop' construct as a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc parallel loop reduction(+:Parm) gang(dim:2)
  for(int i = 0; i < 5; ++i);

  // Reduction cannot appear on a loop with a gang and a num_gangs with >1
  // explicit argument.
#pragma acc kernels loop num_gangs(I) reduction(+:Parm) gang
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop num_gangs(I) gang reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop reduction(+:Parm) num_gangs(I) gang
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop reduction(+:Parm) gang num_gangs(I)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop gang num_gangs(I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop gang reduction(+:Parm) num_gangs(I)
  for(int i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel loop' construct with a 'num_gangs' clause with more than 1 argument}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc parallel loop num_gangs(I, I) reduction(+:Parm) gang
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous 'num_gangs' clause is here}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc parallel loop num_gangs(I, I) gang reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel loop' construct with a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc parallel loop reduction(+:Parm) num_gangs(I, I) gang
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous 'reduction' clause is here}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc parallel loop reduction(+:Parm) gang num_gangs(I, I)
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous 'gang' clause is here}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc parallel loop gang num_gangs(I, I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous 'reduction' clause is here}}
  // expected-note@+1{{previous 'gang' clause is here}}
#pragma acc parallel loop gang reduction(+:Parm) num_gangs(I, I)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel  loop num_gangs(I) reduction(+:Parm) gang
  for(int i = 0; i < 5; ++i);
#pragma acc parallel  loop num_gangs(I) gang reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel  loop reduction(+:Parm) num_gangs(I) gang
  for(int i = 0; i < 5; ++i);
#pragma acc parallel  loop reduction(+:Parm) gang num_gangs(I)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel  loop gang num_gangs(I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel  loop gang reduction(+:Parm) num_gangs(I)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }
#pragma acc parallel loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc parallel reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc parallel loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc parallel loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }
#pragma acc loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc parallel loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc parallel reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous 'reduction' clause is here}}
#pragma acc parallel loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }

  CompositeHasComposite CoCArr[5];
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of array type 'CompositeHasComposite'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel loop reduction(+:CoCArr)
    for(int i = 0; i < 5; ++i);
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of array type 'CompositeHasComposite[5]'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel loop reduction(+:CoCArr[3])
    for(int i = 0; i < 5; ++i);
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of sub-array type 'CompositeHasComposite'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel loop reduction(+:CoCArr[1:1])
    for(int i = 0; i < 5; ++i);
}
