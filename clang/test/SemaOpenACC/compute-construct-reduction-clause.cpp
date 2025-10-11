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
#pragma acc parallel num_gangs(IVar) reduction(+:Var)
  while (1);

  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel' construct with a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc parallel reduction(+:Parm) num_gangs(Parm, IVar)
  while (1);

  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel' construct with a 'num_gangs' clause with more than 1 argument}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc parallel num_gangs(Parm, IVar) reduction(+:Var)
  while (1);

#pragma acc parallel reduction(+:Parm) reduction(+:Parm)
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

  CompositeHasComposite CoCArr[5];
  // expected-error@+4{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of array type 'CompositeHasComposite[5]'}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CoCArr)
  while (1);
  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CoCArr[3])
  while (1);
  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CoCArr[1:1])
  while (1);

  int *IPtr;
  // expected-error@+2{{invalid type 'int *' used in OpenACC 'reduction' variable reference; type is not a scalar value, or array of scalars, or composite of scalars}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:IPtr)
  while (1);
#pragma acc parallel reduction(+:IPtr[1])
  while (1);
#pragma acc parallel reduction(+:IPtr[1:1])
  while (1);

  int *IPtrArr[5];
  // expected-error@+3{{invalid type 'int *' used in OpenACC 'reduction' variable reference; type is not a scalar value, or array of scalars, or composite of scalars}}
  // expected-note@+2{{used as element type of array type 'int *[5]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:IPtrArr)
  while (1);

  struct HasPtr { int *I; }; // #HASPTR
  HasPtr HP;
  // expected-error@+3{{invalid type 'int *' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#HASPTR{{used as field 'I' of composite 'HasPtr'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:HP)
  while (1);

  HasPtr HPArr[5];
  // expected-error@+4{{invalid type 'int *' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+3{{used as element type of array type 'HasPtr[5]'}}
  // expected-note@#HASPTR{{used as field 'I' of composite 'HasPtr'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:HPArr)
  while (1);

  _Complex int CplxI;
  _Complex int CplxIArr[5];
  _Complex float CplxF;
  _Complex float CplxFArr[5];
  struct HasCplx { _Complex int I; } HC; //#HASCPLX
  // expected-error@+2{{invalid type '_Complex int' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CplxI)
  while (1);
  // expected-error@+3{{invalid type '_Complex int' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+2{{used as element type of array type '_Complex int[5]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CplxIArr)
  while (1);
  // expected-error@+2{{invalid type '_Complex float' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CplxF)
  while (1);
  // expected-error@+3{{invalid type '_Complex float' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+2{{used as element type of array type '_Complex float[5]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:CplxFArr)
  while (1);
  // expected-error@+3{{invalid type '_Complex int' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#HASCPLX{{used as field 'I' of composite 'HasCplx'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:HC)
  while (1);
}

template<typename T, typename U, typename V>
void TemplUses(T Parm, U CoS, V ChC) {
  T Var;
  U *CoSPtr;
  V *ChCPtr;

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
#pragma acc parallel reduction(+:Parm) num_gangs(Var)
  while (1);
#pragma acc parallel num_gangs(Var) reduction(+:Var)
  while (1);

  // expected-error@+2{{OpenACC 'num_gangs' clause with more than 1 argument may not appear on a 'parallel' construct with a 'reduction' clause}}
  // expected-note@+1{{previous 'reduction' clause is here}}
#pragma acc parallel reduction(+:Parm) num_gangs(Parm, Var)
  while (1);

  // expected-error@+2{{OpenACC 'reduction' clause may not appear on a 'parallel' construct with a 'num_gangs' clause with more than 1 argument}}
  // expected-note@+1{{previous 'num_gangs' clause is here}}
#pragma acc parallel num_gangs(Parm, Var) reduction(+:Var)
  while (1);

#pragma acc parallel reduction(+:Parm) reduction(+:Parm)
  while (1);

  int NonDep;
  int NonDepArray[5];
  T Array[5];

  // Vars in a reduction must be a scalar or a composite of scalars.
#pragma acc parallel reduction(&: CoS, Var, Parm)
  while (1);
  // expected-error@+3{{invalid type 'struct CompositeOfScalars' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@#COS_FIELD{{used as field 'COS' of composite 'CompositeHasComposite'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(&: ChC)
  while (1);
#pragma acc parallel reduction(&: Array)
  while (1);
#pragma acc parallel reduction(&: NonDepArray)
  while (1);

#pragma acc parallel reduction(&: CoS, Array[Var], Array[0:Var])
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

  T ThreeDArray[3][4][5];

  // expected-error@+3{{invalid type 'int[4][5]' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+2{{used as element type of array type 'int[3][4][5]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:ThreeDArray)
  while (1);
  // expected-error@+3{{invalid type 'int[5]' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+2{{used as element type of array type 'int[4][5]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:ThreeDArray[1:1])
  while (1);
  // expected-error@+3{{invalid type 'int[5]' used in OpenACC 'reduction' variable reference; type is not a scalar value}}
  // expected-note@+2{{used as element type of array type 'int[4][5]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+:ThreeDArray[1])
  while (1);

#pragma acc parallel reduction(+:ThreeDArray[1:1][1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1:1][1:1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1][1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1][1:1])
  while (1);

#pragma acc parallel reduction(+:ThreeDArray[1:1][1][1:1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1:1][1][1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1:1][1:1][1:1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1:1][1:1][1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1][1][1:1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1][1][1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1][1:1][1:1])
  while (1);
#pragma acc parallel reduction(+:ThreeDArray[1][1:1][1])
  while (1);
}

void inst() {
  CompositeOfScalars CoS;
  CompositeHasComposite ChC;
  // expected-note@+1{{in instantiation of function template specialization}}
  TemplUses(5, CoS, ChC);
}
