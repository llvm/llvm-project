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

  // expected-error@+2{{OpenACC 'reduction' composite variable must not have non-scalar field}}
  // expected-note@#COS_FIELD{{invalid field is here}}
#pragma acc parallel loop reduction(&: ChC)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop reduction(+:Parm) num_gangs(I)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel loop' construct with a 'reduction' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop reduction(+:Parm) num_gangs(I, I)
  for(int i = 0; i < 5; ++i);

#pragma acc kernels loop num_gangs(I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel loop' construct with a 'num_gangs' clause with more than 1 argument}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop num_gangs(I, I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);

  // Reduction cannot appear on a loop with a 'gang' of dim>1.
#pragma acc parallel loop gang(dim:1) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause with a 'dim' value greater than 1}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop gang(dim:2) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop reduction(+:Parm) gang(dim:1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'gang' clause with a 'dim' value greater than 1 cannot appear on the same 'parallel loop' construct as a 'reduction' clause}}
  // expected-note@+1{{previous clause is here}}
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
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop num_gangs(I, I) reduction(+:Parm) gang
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous clause is here}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop num_gangs(I, I) gang reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel loop' construct with a 'reduction' clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop reduction(+:Parm) num_gangs(I, I) gang
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous clause is here}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop reduction(+:Parm) gang num_gangs(I, I)
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous clause is here}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop gang num_gangs(I, I) reduction(+:Parm)
  for(int i = 0; i < 5; ++i);
  // expected-error@+3{{OpenACC 'reduction' clause cannot appear on the same 'parallel loop' construct as a 'gang' clause and a 'num_gangs' clause with more than one argument}}
  // expected-note@+2{{previous clause is here}}
  // expected-note@+1{{previous clause is here}}
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
  // expected-note@-3{{previous clause is here}}
#pragma acc loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }
#pragma acc parallel loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous clause is here}}
#pragma acc parallel reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc parallel loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous clause is here}}
#pragma acc parallel loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }
#pragma acc loop reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous clause is here}}
#pragma acc parallel loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }

#pragma acc parallel reduction(+:I)
  for(int i = 0; i < 5; ++i) {
  // expected-error@+2{{OpenACC 'reduction' variable must have the same operator in all nested constructs (& vs +)}}
  // expected-note@-3{{previous clause is here}}
#pragma acc parallel loop reduction(&:I)
    for(int i = 0; i < 5; ++i);
  }
}
