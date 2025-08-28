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

  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel' construct with a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc parallel reduction(+:Parm) num_gangs(Parm, IVar)
  while (1);

  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel' construct with a 'num_gangs' clause with more than 1 argument}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
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
  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(&: ChC)
  while (1);

#pragma acc parallel reduction(&: Array)
  while (1);

#pragma acc parallel reduction(&: CoS, Array[I], Array[0:I])
  while (1);

  struct CompositeHasComposite ChCArray[5];
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of sub-array type 'struct CompositeHasComposite'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
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

  // expected-error@+1{{OpenACC 'reduction' clause is not valid on 'init' directive}}
#pragma acc init reduction(+:I)
  for(;;);
}
