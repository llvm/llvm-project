// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
int getI();
void uses() {
  int Var;
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update async self(Var)
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update wait self(Var)
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update self(Var) device_type(I)
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update if(true) self(Var)
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update if_present self(Var)
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update self(Var)
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc update host(Var)
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)

  // expected-warning@+3{{OpenACC clause 'self' not yet implemented}}
  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) if(true)
  // expected-warning@+3{{OpenACC clause 'self' not yet implemented}}
  // expected-error@+2{{OpenACC clause 'if_present' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update self(Var) device_type(I) if_present
  // expected-error@+2{{OpenACC clause 'self' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update device_type(I) self(Var)
  // expected-error@+2{{OpenACC clause 'host' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update device_type(I) host(Var)
  // expected-error@+2{{OpenACC clause 'device' may not follow a 'device_type' clause in a 'update' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update device_type(I) device(Var)
  // These 2 are OK.
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update self(Var) device_type(I) async
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update self(Var) device_type(I) wait
  // Unless otherwise specified, we assume 'device_type' can happen after itself.
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update self(Var) device_type(I) device_type(I)

  // TODO: OpenACC: These should diagnose because there isn't at least 1 of
  // 'self', 'host', or 'device'.
#pragma acc update async
#pragma acc update wait
#pragma acc update device_type(I)
#pragma acc update if(true)
#pragma acc update if_present

  // expected-error@+1{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
#pragma acc update if (NC) device_type(I)

  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'update' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update if(true) if (false)

  // TODO: OpenACC: There is restrictions on the contents of a 'varlist', so
  // those should be checked here too.

  // Cannot be the body of an 'if', 'while', 'do', 'switch', or
  // 'label'.
  // expected-error@+3{{OpenACC 'update' construct may not appear in place of the statement following an if statement}}
  if (true)
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)

  // expected-error@+3{{OpenACC 'update' construct may not appear in place of the statement following a while statement}}
  while (true)
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)

  // expected-error@+3{{OpenACC 'update' construct may not appear in place of the statement following a do statement}}
  do
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)
  while (true);

  // expected-error@+3{{OpenACC 'update' construct may not appear in place of the statement following a switch statement}}
  switch(Var)
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)

  // expected-error@+3{{OpenACC 'update' construct may not appear in place of the statement following a label statement}}
  LABEL:
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)

  // For loops are OK.
  for (;;)
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device(Var)

  // Checking for 'async', which requires an 'int' expression.
#pragma acc update async

#pragma acc update async(getI())
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc update async(getI(), getI())
  // expected-error@+2{{OpenACC 'async' clause cannot appear more than once on a 'update' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc update async(getI()) async(getI())
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update async(NC)

  // Checking for 'wait', which has a complicated set arguments.
#pragma acc update wait
#pragma acc update wait()
#pragma acc update wait(getI(), getI())
#pragma acc update wait(devnum: getI():  getI())
#pragma acc update wait(devnum: getI(): queues: getI(), getI())
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update wait(devnum:NC : 5)
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update wait(devnum:5 : NC)

    int arr[5];
  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc update wait(devnum:arr : queues: arr, NC, 5)
}
