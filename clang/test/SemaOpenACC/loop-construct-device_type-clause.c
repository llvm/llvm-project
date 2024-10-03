// RUN: %clang_cc1 %s -fopenacc -verify

#define MACRO +FOO

void uses() {
  typedef struct S{} STy;
  STy SImpl;

#pragma acc loop device_type(I)
  for(;;);
#pragma acc loop device_type(S) dtype(STy)
  for(;;);
#pragma acc loop dtype(SImpl)
  for(;;);
#pragma acc loop dtype(int) device_type(*)
  for(;;);
#pragma acc loop dtype(true) device_type(false)
  for(;;);

  // expected-error@+1{{expected identifier}}
#pragma acc loop dtype(int, *)
  for(;;);

#pragma acc loop device_type(I, int)
  for(;;);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc loop dtype(int{})
  for(;;);
  // expected-error@+1{{expected identifier}}
#pragma acc loop dtype(5)
  for(;;);
  // expected-error@+1{{expected identifier}}
#pragma acc loop dtype(MACRO)
  for(;;);


  // Only 'collapse', 'gang', 'worker', 'vector', 'seq', 'independent', 'auto',
  // and 'tile'  allowed after 'device_type'.

  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented, clause ignored}}
#pragma acc loop device_type(*) vector
  for(;;);

  // expected-error@+2{{OpenACC clause 'finalize' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) finalize
  for(;;);
  // expected-error@+2{{OpenACC clause 'if_present' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) if_present
  for(;;);
#pragma acc loop device_type(*) seq
  for(;;);
#pragma acc loop device_type(*) independent
  for(;;);
#pragma acc loop device_type(*) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented, clause ignored}}
#pragma acc loop device_type(*) worker
  for(;;);
  // expected-error@+2{{OpenACC clause 'nohost' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) nohost
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) default(none)
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) if(1)
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) self
  for(;;);

  int Var;
  int *VarPtr;
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) copy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) pcopy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) present_or_copy(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'use_device' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) use_device(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) attach(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'delete' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) delete(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'detach' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) detach(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'device' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) device(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) deviceptr(VarPtr)
  for(;;);
  // expected-error@+2{{OpenACC clause 'device_resident' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*)  device_resident(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) firstprivate(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'host' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) host(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'link' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) link(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) no_create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) present(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'private' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) private(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) pcopyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) present_or_copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) pcopyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) present_or_copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) pcreate(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) present_or_create(Var)
  for(;;);
  // expected-error@+2{{OpenACC clause 'reduction' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) reduction(+:Var)
  for(;;);
#pragma acc loop device_type(*) collapse(1)
  for(;;);
  // expected-error@+2{{OpenACC clause 'bind' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) bind(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) vector_length(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) num_gangs(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) num_workers(1)
  for(;;);
  // expected-error@+2{{OpenACC clause 'device_num' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) device_num(1)
  for(;;);
  // expected-error@+2{{OpenACC clause 'default_async' may not follow a 'device_type' clause in a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop device_type(*) default_async(1)
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) async
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented, clause ignored}}
#pragma acc loop device_type(*) tile(Var, 1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented, clause ignored}}
#pragma acc loop dtype(*) gang
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop device_type(*) wait
  for(;;);
}
