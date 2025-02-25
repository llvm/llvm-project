// RUN: %clang_cc1 %s -fopenacc -verify

#define MACRO +FOO

void uses() {
  typedef struct S{} STy;
  STy SImpl;

#pragma acc parallel loop device_type(I)
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop device_type(S) dtype(STy)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop dtype(SImpl)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop dtype(int) device_type(*)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop dtype(true) device_type(false)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected identifier}}
#pragma acc kernels loop dtype(int, *)
  for(int i = 0; i < 5; ++i);

#pragma acc parallel loop device_type(I, int)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc kernels loop dtype(int{})
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected identifier}}
#pragma acc kernels loop dtype(5)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected identifier}}
#pragma acc kernels loop dtype(MACRO)
  for(int i = 0; i < 5; ++i);

  // Compute constructs allow 'async', 'wait', num_gangs', 'num_workers',
  // 'vector_length' after 'device_type', loop allows 'collapse', 'gang',
  // 'worker', 'vector', 'seq', 'independent', 'auto', and 'tile'  after
  // 'device_type'.

#pragma acc parallel loop device_type(*) vector
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) finalize
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'kernels loop' directive}}
#pragma acc kernels loop device_type(*) if_present
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) seq
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop device_type(*) independent
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop device_type(*) auto
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) worker
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'nohost' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) nohost
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'default' may not follow a 'device_type' clause in a 'kernels loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop device_type(*) default(none)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) if(1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'self' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) self
  for(int i = 0; i < 5; ++i);

  int Var;
  int *VarPtr;
  // expected-error@+2{{OpenACC clause 'copy' may not follow a 'device_type' clause in a 'kernels loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop device_type(*) copy(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'pcopy' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) pcopy(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'present_or_copy' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) present_or_copy(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'kernels loop' directive}}
#pragma acc kernels loop device_type(*) use_device(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'attach' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) attach(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) delete(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'kernels loop' directive}}
#pragma acc kernels loop device_type(*) detach(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device_type(*) device(VarPtr)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'deviceptr' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) deviceptr(VarPtr)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'device_resident' may not follow a 'device_type' clause in a 'kernels loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop device_type(*)  device_resident(VarPtr)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'firstprivate' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) firstprivate(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) host(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'link' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) link(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'no_create' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) no_create(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'present' may not follow a 'device_type' clause in a 'kernels loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc kernels loop device_type(*) present(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'private' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) private(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'copyout' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) copyout(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'pcopyout' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) pcopyout(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'present_or_copyout' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) present_or_copyout(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'copyin' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) copyin(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'pcopyin' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) pcopyin(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'present_or_copyin' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) present_or_copyin(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'create' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) create(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'pcreate' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) pcreate(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'present_or_create' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) present_or_create(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'reduction' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop device_type(*) reduction(+:Var)
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop device_type(*) collapse(1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'bind' may not follow a 'device_type' clause in a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop device_type(*) bind(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) vector_length(1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) num_gangs(1)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) num_workers(1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) device_num(1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) default_async(1)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) async
  for(int i = 0; i < 5; ++i);

#pragma acc serial loop device_type(*) tile(*, 1)
  for(int j = 0; j < 5; ++j)
    for(int i = 0; i < 5; ++i);

#pragma acc serial loop dtype(*) gang
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) wait
  for(int i = 0; i < 5; ++i);
}
