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

  //expected-warning@+1{{OpenACC clause 'vector' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) vector
  for(int i = 0; i < 5; ++i);

  // TODOexpected-error@+2{{OpenACC clause 'finalize' may not follow a 'device_type' clause in a 'serial loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) finalize
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'if_present' may not follow a 'device_type' clause in a 'kernels loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*) if_present
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) seq
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop device_type(*) independent
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop device_type(*) auto
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) worker
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'nohost' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) nohost
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'default' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*) default(none)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'if' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) if(1)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'self' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) self
  for(int i = 0; i < 5; ++i);

  int Var;
  int *VarPtr;
  // expected-warning@+1{{OpenACC clause 'copy' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*) copy(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
  // expected-warning@+1{{OpenACC clause 'pcopy' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) pcopy(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
  // expected-warning@+1{{OpenACC clause 'present_or_copy' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) present_or_copy(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'use_device' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*) use_device(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'attach' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) attach(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'delete' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) delete(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'detach' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*) detach(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'device' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) device(VarPtr)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'deviceptr' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) deviceptr(VarPtr)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'device_resident' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*)  device_resident(VarPtr)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'firstprivate' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) firstprivate(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'host' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) host(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'link' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) link(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'no_create' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) no_create(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'present' not yet implemented, clause ignored}}
#pragma acc kernels loop device_type(*) present(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'private' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'private' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) private(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'copyout' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) copyout(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
  // expected-warning@+1{{OpenACC clause 'pcopyout' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) pcopyout(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
  // expected-warning@+1{{OpenACC clause 'present_or_copyout' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) present_or_copyout(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'copyin' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) copyin(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
  // expected-warning@+1{{OpenACC clause 'pcopyin' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) pcopyin(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
  // expected-warning@+1{{OpenACC clause 'present_or_copyin' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) present_or_copyin(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'create' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) create(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
  // expected-warning@+1{{OpenACC clause 'pcreate' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) pcreate(Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+2{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
  // expected-warning@+1{{OpenACC clause 'present_or_create' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) present_or_create(Var)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'reduction' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) reduction(+:Var)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) collapse(1)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'bind' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) bind(Var)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) vector_length(1)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial loop' directive}}
#pragma acc serial loop device_type(*) num_gangs(1)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'num_workers' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) num_workers(1)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'device_num' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) device_num(1)
  for(int i = 0; i < 5; ++i);
  // TODOexpected-error@+2{{OpenACC clause 'default_async' may not follow a 'device_type' clause in a 'loop' construct}}
  // TODOexpected-note@+1{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) default_async(1)
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'async' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) async
  for(int i = 0; i < 5; ++i);

  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented, clause ignored}}
#pragma acc serial loop device_type(*) tile(*, 1)
  for(int j = 0; j < 5; ++j)
    for(int i = 0; i < 5; ++i);

  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented, clause ignored}}
#pragma acc serial loop dtype(*) gang
  for(int i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'wait' not yet implemented, clause ignored}}
#pragma acc parallel loop device_type(*) wait
  for(int i = 0; i < 5; ++i);
}
