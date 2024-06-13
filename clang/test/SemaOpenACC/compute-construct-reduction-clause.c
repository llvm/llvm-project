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

void uses(unsigned Parm) {
  float Var;
  int IVar;

#pragma acc parallel reduction(+:Parm)
  while (1);
#pragma acc serial reduction(+:Parm)
  while (1);
  // expected-error@+1{{OpenACC 'reduction' clause is not valid on 'kernels' directive}}
#pragma acc kernels reduction(+:Parm)
  while (1);

  // On a 'parallel', 'num_gangs' cannot have >1 args. num_gangs not valid on
  // 'serial', but 'reduction' not valid on 'kernels', other combos cannot be
  // tested.
#pragma acc parallel reduction(+:Parm) num_gangs(IVar)
  while (1);
#pragma acc parallel num_gangs(IVar) reduction(+:IVar)
  while (1);

  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel' construct with a 'num_gangs' clause with more than 1 argument, have 2}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel reduction(+:Parm) num_gangs(Parm, IVar)
  while (1);

  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel' construct with a 'num_gangs' clause with more than 1 argument, have 2}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel num_gangs(Parm, IVar) reduction(+:Var)
  while (1);

  struct CompositeOfScalars CoS;
  struct CompositeOfScalars *CoSPtr;
  struct CompositeHasComposite ChC;
  struct CompositeHasComposite *ChCPtr;

  int I;
  float F;
  int Array[5];

  // Vars in a reduction must be a scalar or a composite of scalars.
#pragma acc parallel reduction(&: CoS, I, F)
  while (1);
  // expected-error@+2{{OpenACC 'reduction' composite variable must not have non-scalar field}}
  // expected-note@#COS_FIELD{{invalid field is here}}
#pragma acc parallel reduction(&: ChC)
  while (1);

  // expected-error@+1{{OpenACC 'reduction' variable must be of scalar type, sub-array, or a composite of scalar types; type is 'int[5]'}}
#pragma acc parallel reduction(&: Array)
  while (1);

#pragma acc parallel reduction(&: CoS, Array[I], Array[0:I])
  while (1);

  struct CompositeHasComposite ChCArray[5];
  // expected-error@+1{{OpenACC 'reduction' variable must be of scalar type, sub-array, or a composite of scalar types; sub-array base type is 'struct CompositeHasComposite'}}
#pragma acc parallel reduction(&: CoS, Array[I], ChCArray[0:I])
  while (1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel reduction(&: CoS.I)
  while (1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel reduction(&: CoSPtr->I)

  while (1);
  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel reduction(&: ChC.COS)
  while (1);

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel reduction(&: ChCPtr->COS)
  while (1);

#pragma acc parallel reduction(&: I) reduction(&:I)
  while (1);

  struct HasArray { int array[5]; } HA;

  // expected-error@+1{{OpenACC variable is not a valid variable name, sub-array, array element, or composite variable member}}
#pragma acc parallel reduction(&:HA.array[1:2])
  while (1);

  // expected-error@+2{{OpenACC 'reduction' clause is not valid on 'init' directive}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented}}
#pragma acc init reduction(+:I)
  for(;;);
}
