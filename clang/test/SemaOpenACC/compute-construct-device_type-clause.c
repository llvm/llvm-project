// RUN: %clang_cc1 %s -fopenacc -verify

#define MACRO +FOO

void uses() {
  typedef struct S{} STy;
  STy SImpl;

#pragma acc parallel device_type(I)
  while(1);
#pragma acc serial device_type(S) dtype(STy)
  while(1);
#pragma acc kernels dtype(SImpl)
  while(1);
#pragma acc kernels dtype(int) device_type(*)
  while(1);
#pragma acc kernels dtype(true) device_type(false)
  while(1);

  // expected-error@+1{{expected identifier}}
#pragma acc kernels dtype(int, *)
  while(1);

#pragma acc parallel device_type(I, int)
  while(1);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc kernels dtype(int{})
  while(1);
  // expected-error@+1{{expected identifier}}
#pragma acc kernels dtype(5)
  while(1);
  // expected-error@+1{{expected identifier}}
#pragma acc kernels dtype(MACRO)
  while(1);

  // expected-error@+2{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
  // expected-error@+1{{OpenACC 'device_type' clause is not valid on 'enter data' directive}}
#pragma acc enter data device_type(I)
  // expected-error@+2{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
  // expected-error@+1{{OpenACC 'dtype' clause is not valid on 'enter data' directive}}
#pragma acc enter data dtype(I)


  // Only 'async', 'wait', num_gangs', 'num_workers', 'vector_length' allowed after 'device_type'.

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) finalize
  while(1);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) if_present
  while(1);
  // expected-error@+1{{OpenACC 'seq' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) seq
  while(1);
  // expected-error@+1{{OpenACC 'independent' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) independent
  while(1);
  // expected-error@+1{{OpenACC 'auto' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) auto
  while(1);
  // expected-error@+1{{OpenACC 'worker' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) worker
  while(1);
  // expected-error@+2{{OpenACC clause 'nohost' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) nohost
  while(1);
  // expected-error@+2{{OpenACC clause 'default' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) default(none)
  while(1);
  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) if(1)
  while(1);
  // expected-error@+2{{OpenACC clause 'self' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) self
  while(1);

  int Var;
  int *VarPtr;
  // expected-error@+2{{OpenACC clause 'copy' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) copy(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'pcopy' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) pcopy(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'present_or_copy' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) present_or_copy(Var)
  while(1);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) use_device(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'attach' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) attach(Var)
  while(1);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) delete(Var)
  while(1);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) detach(Var)
  while(1);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) device(VarPtr)
  while(1);
  // expected-error@+2{{OpenACC clause 'deviceptr' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) deviceptr(VarPtr)
  while(1);
  // expected-error@+2{{OpenACC clause 'device_resident' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*)  device_resident(VarPtr)
  while(1);
  // expected-error@+2{{OpenACC clause 'firstprivate' may not follow a 'device_type' clause in a 'parallel' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel device_type(*) firstprivate(Var)
  while(1);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) host(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'link' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) link(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'no_create' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) no_create(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'present' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) present(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'private' may not follow a 'device_type' clause in a 'parallel' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel device_type(*) private(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'copyout' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) copyout(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'pcopyout' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) pcopyout(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'present_or_copyout' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) present_or_copyout(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'copyin' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) copyin(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'pcopyin' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) pcopyin(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'present_or_copyin' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) present_or_copyin(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'create' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) create(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'pcreate' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) pcreate(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'present_or_create' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) present_or_create(Var)
  while(1);
  // expected-error@+2{{OpenACC clause 'reduction' may not follow a 'device_type' clause in a 'serial' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial device_type(*) reduction(+:Var)
  while(1);
  // expected-error@+1{{OpenACC 'collapse' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) collapse(1)
  while(1);
  // expected-error@+2{{OpenACC clause 'bind' may not follow a 'device_type' clause in a 'kernels' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels device_type(*) bind(Var)
  while(1);
#pragma acc kernels device_type(*) vector_length(1)
  while(1);
#pragma acc kernels device_type(*) num_gangs(1)
  while(1);
#pragma acc kernels device_type(*) num_workers(1)
  while(1);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) device_num(1)
  while(1);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) default_async(1)
  while(1);
#pragma acc kernels device_type(*) async
  while(1);
  // expected-error@+1{{OpenACC 'tile' clause is not valid on 'kernels' directive}}
#pragma acc kernels device_type(*) tile(Var, 1)
  while(1);
  // expected-error@+1{{OpenACC 'gang' clause is not valid on 'kernels' directive}}
#pragma acc kernels dtype(*) gang
  while(1);
#pragma acc kernels device_type(*) wait
  while(1);
}
