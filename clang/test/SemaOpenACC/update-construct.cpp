// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
void uses() {
  int Var;
  // expected-warning@+2{{OpenACC clause 'async' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update async self(Var)
  // expected-warning@+2{{OpenACC clause 'wait' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update wait self(Var)
  // expected-warning@+2{{OpenACC clause 'self' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'device_type' not yet implemented}}
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

  // TODO: OpenACC: These all should diagnose as they aren't allowed after
  // device_type.
    // expected-warning@+3{{OpenACC clause 'self' not yet implemented}}
    // expected-warning@+2{{OpenACC clause 'device_type' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'device_type' not yet implemented}}
#pragma acc update self(Var) device_type(I) device_type(I)
    // expected-warning@+2{{OpenACC clause 'self' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'device_type' not yet implemented}}
#pragma acc update self(Var) device_type(I) if(true)
    // expected-warning@+2{{OpenACC clause 'self' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'device_type' not yet implemented}}
#pragma acc update self(Var) device_type(I) if_present
    // expected-warning@+2{{OpenACC clause 'device_type' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'self' not yet implemented}}
#pragma acc update device_type(I) self(Var)
    // expected-warning@+2{{OpenACC clause 'device_type' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc update device_type(I) host(Var)
    // expected-warning@+2{{OpenACC clause 'device_type' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc update device_type(I) device(Var)
  // These 2 are OK.
    // expected-warning@+3{{OpenACC clause 'self' not yet implemented}}
    // expected-warning@+2{{OpenACC clause 'device_type' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'async' not yet implemented}}
#pragma acc update self(Var) device_type(I) async
    // expected-warning@+3{{OpenACC clause 'self' not yet implemented}}
    // expected-warning@+2{{OpenACC clause 'device_type' not yet implemented}}
    // expected-warning@+1{{OpenACC clause 'wait' not yet implemented}}
#pragma acc update self(Var) device_type(I) wait

  // TODO: OpenACC: These should diagnose because there isn't at least 1 of
  // 'self', 'host', or 'device'.
    // expected-warning@+1{{OpenACC clause 'async' not yet implemented}}
#pragma acc update async
    // expected-warning@+1{{OpenACC clause 'wait' not yet implemented}}
#pragma acc update wait
    // expected-warning@+1{{OpenACC clause 'device_type' not yet implemented}}
#pragma acc update device_type(I)
#pragma acc update if(true)
#pragma acc update if_present

  // expected-error@+2{{value of type 'struct NotConvertible' is not contextually convertible to 'bool'}}
  // expected-warning@+1{{OpenACC clause 'device_type' not yet implemented}}
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
}
