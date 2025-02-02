// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
int getI();
void uses() {
  int Var;
#pragma acc update async self(Var)
#pragma acc update wait self(Var)
#pragma acc update self(Var) device_type(I)
#pragma acc update if(true) self(Var)
#pragma acc update if_present self(Var)
#pragma acc update self(Var)
#pragma acc update host(Var)
#pragma acc update device(Var)

  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) if(true)
  // expected-error@+2{{OpenACC clause 'if_present' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) if_present
  // expected-error@+2{{OpenACC clause 'self' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) self(Var)
  // expected-error@+2{{OpenACC clause 'host' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) host(Var)
  // expected-error@+2{{OpenACC clause 'device' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) device(Var)
  // These 2 are OK.
#pragma acc update self(Var) device_type(I) async
#pragma acc update self(Var) device_type(I) wait
  // Unless otherwise specified, we assume 'device_type' can happen after itself.
#pragma acc update self(Var) device_type(I) device_type(I)

  // These diagnose because there isn't at least 1 of 'self', 'host', or
  // 'device'.
  // expected-error@+1{{OpenACC 'update' construct must have at least one 'self', 'host' or 'device' clause}}
#pragma acc update async
  // expected-error@+1{{OpenACC 'update' construct must have at least one 'self', 'host' or 'device' clause}}
#pragma acc update wait
  // expected-error@+1{{OpenACC 'update' construct must have at least one 'self', 'host' or 'device' clause}}
#pragma acc update device_type(I)
  // expected-error@+1{{OpenACC 'update' construct must have at least one 'self', 'host' or 'device' clause}}
#pragma acc update if(true)
  // expected-error@+1{{OpenACC 'update' construct must have at least one 'self', 'host' or 'device' clause}}
#pragma acc update if_present

  // expected-error@+1{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc update self(Var) if (NC) device_type(I)

  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'update' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) if(true) if (false)

  // Cannot be the body of an 'if', 'while', 'do', 'switch', or
  // 'label'.
  // expected-error@+2{{OpenACC 'update' construct may not appear in place of the statement following an if statement}}
  if (true)
#pragma acc update device(Var)

  // expected-error@+2{{OpenACC 'update' construct may not appear in place of the statement following a while statement}}
  while (true)
#pragma acc update device(Var)

  // expected-error@+2{{OpenACC 'update' construct may not appear in place of the statement following a do statement}}
  do
#pragma acc update device(Var)
  while (true);

  // expected-error@+2{{OpenACC 'update' construct may not appear in place of the statement following a switch statement}}
  switch(Var)
#pragma acc update device(Var)

  // expected-error@+2{{OpenACC 'update' construct may not appear in place of the statement following a label statement}}
  LABEL:
#pragma acc update device(Var)

  // For loops are OK.
  for (;;)
#pragma acc update device(Var)

  // Checking for 'async', which requires an 'int' expression.
#pragma acc update self(Var) async

#pragma acc update self(Var) async(getI())
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc update self(Var) async(getI(), getI())
  // expected-error@+2{{OpenACC 'async' clause cannot appear more than once on a 'update' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) async(getI()) async(getI())
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update self(Var) async(NC)

  // Checking for 'wait', which has a complicated set arguments.
#pragma acc update self(Var) wait
#pragma acc update self(Var) wait()
#pragma acc update self(Var) wait(getI(), getI())
#pragma acc update self(Var) wait(devnum: getI():  getI())
#pragma acc update self(Var) wait(devnum: getI(): queues: getI(), getI())
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update self(Var) wait(devnum:NC : 5)
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update self(Var) wait(devnum:5 : NC)

    int arr[5];
  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update self(Var) wait(devnum:arr : queues: arr, NC, 5)
}

struct SomeS {
  int Array[5];
  int MemberOfComp;
};

template<typename I, typename T>
void varlist_restrictions_templ() {
  I iArray[5];
  T Single;
  T Array[5];

  // Members of a subarray of struct or class type may not appear, but others
  // are permitted to.
#pragma acc update self(iArray[0:1])
#pragma acc update host(iArray[0:1])
#pragma acc update device(iArray[0:1])

#pragma acc update self(Array[0:1])
#pragma acc update host(Array[0:1])
#pragma acc update device(Array[0:1])

  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc update self(Array[0:1].MemberOfComp)
  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc update host(Array[0:1].MemberOfComp)
  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc update device(Array[0:1].MemberOfComp)
}

void varlist_restrictions() {
  varlist_restrictions_templ<int, SomeS>();// expected-note{{in instantiation of}}
  int iArray[5];
  SomeS Single;
  SomeS Array[5];

  int LocalInt;
  int *LocalPtr;

#pragma acc update self(LocalInt, LocalPtr, Single)
#pragma acc update host(LocalInt, LocalPtr, Single)
#pragma acc update device(LocalInt, LocalPtr, Single)

#pragma acc update self(Single.MemberOfComp)
#pragma acc update host(Single.MemberOfComp)
#pragma acc update device(Single.MemberOfComp)

#pragma acc update self(Single.Array[0:1])
#pragma acc update host(Single.Array[0:1])
#pragma acc update device(Single.Array[0:1])


  // Members of a subarray of struct or class type may not appear, but others
  // are permitted to.
#pragma acc update self(iArray[0:1])
#pragma acc update host(iArray[0:1])
#pragma acc update device(iArray[0:1])

#pragma acc update self(Array[0:1])
#pragma acc update host(Array[0:1])
#pragma acc update device(Array[0:1])

  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc update self(Array[0:1].MemberOfComp)
  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc update host(Array[0:1].MemberOfComp)
  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc update device(Array[0:1].MemberOfComp)
}

