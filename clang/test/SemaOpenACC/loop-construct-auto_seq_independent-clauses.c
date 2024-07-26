// RUN: %clang_cc1 %s -fopenacc -verify

void uses() {
#pragma acc loop auto
  for(;;);
#pragma acc loop seq
  for(;;);
#pragma acc loop independent
  for(;;);

  // expected-error@+2{{OpenACC clause 'seq' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop auto seq
  for(;;);
  // expected-error@+2{{OpenACC clause 'independent' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop auto independent
  for(;;);
  // expected-error@+2{{OpenACC clause 'auto' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq auto
  for(;;);
  // expected-error@+2{{OpenACC clause 'independent' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq independent
  for(;;);
  // expected-error@+2{{OpenACC clause 'auto' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop independent auto
  for(;;);
  // expected-error@+2{{OpenACC clause 'seq' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop independent seq
  for(;;);

  int Var;
  int *VarPtr;

  // 'auto' can combine with any other clause.
  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc loop auto finalize
  for(;;);
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc loop auto if_present
  for(;;);
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented}}
#pragma acc loop auto worker
  for(;;);
  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented}}
#pragma acc loop auto vector
  for(;;);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop auto nohost
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop auto default(none)
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop auto if(1)
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop auto self
  for(;;);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop auto copy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcopy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_copy(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc loop auto use_device(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop auto attach(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc loop auto delete(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc loop auto detach(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc loop auto device(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop auto deviceptr(VarPtr)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop auto device_resident(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop auto firstprivate(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc loop auto host(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop auto link(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop auto no_create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop auto present(Var)
  for(;;);
#pragma acc loop auto private(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop auto copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcopyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop auto copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcopyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop auto create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcreate(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_create(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented}}
#pragma acc loop auto reduction(+:Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented}}
#pragma acc loop auto collapse(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop auto bind(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop auto vector_length(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop auto num_gangs(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop auto num_workers(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented}}
#pragma acc loop auto device_num(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented}}
#pragma acc loop auto default_async(1)
  for(;;);
#pragma acc loop auto device_type(*)
  for(;;);
#pragma acc loop auto dtype(*)
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop auto async
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented}}
#pragma acc loop auto tile(Var, 1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented}}
#pragma acc loop auto gang
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop auto wait
  for(;;);

  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc loop finalize auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc loop if_present auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented}}
#pragma acc loop worker auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented}}
#pragma acc loop vector auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop nohost auto
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(none) auto
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(1) auto
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self auto
  for(;;);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop copy(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop pcopy(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copy(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc loop use_device(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc loop delete(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc loop detach(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc loop device(VarPtr) auto
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(VarPtr) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop device_resident(VarPtr) auto
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop firstprivate(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc loop host(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop link(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop present(Var) auto
  for(;;);
#pragma acc loop private(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop create(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop pcreate(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_create(Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented}}
#pragma acc loop reduction(+:Var) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented}}
#pragma acc loop collapse(1) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop bind(Var) auto
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1) auto
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1) auto
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented}}
#pragma acc loop device_num(1) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented}}
#pragma acc loop default_async(1) auto
  for(;;);
#pragma acc loop device_type(*) auto
  for(;;);
#pragma acc loop dtype(*) auto
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented}}
#pragma acc loop tile(Var, 1) auto
  for(;;);
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented}}
#pragma acc loop gang auto
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait auto
  for(;;);

  // 'independent' can also be combined with any clauses
  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc loop independent finalize
  for(;;);
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc loop independent if_present
  for(;;);
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented}}
#pragma acc loop independent worker
  for(;;);
  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented}}
#pragma acc loop independent vector
  for(;;);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop independent nohost
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop independent default(none)
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop independent if(1)
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop independent self
  for(;;);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop independent copy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcopy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_copy(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc loop independent use_device(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop independent attach(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc loop independent delete(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc loop independent detach(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc loop independent device(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop independent deviceptr(VarPtr)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop independent device_resident(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop independent firstprivate(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc loop independent host(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop independent link(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop independent no_create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop independent present(Var)
  for(;;);
#pragma acc loop independent private(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop independent copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcopyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop independent copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcopyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop independent create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcreate(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_create(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented}}
#pragma acc loop independent reduction(+:Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented}}
#pragma acc loop independent collapse(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop independent bind(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop independent vector_length(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop independent num_gangs(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop independent num_workers(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented}}
#pragma acc loop independent device_num(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented}}
#pragma acc loop independent default_async(1)
  for(;;);
#pragma acc loop independent device_type(*)
  for(;;);
#pragma acc loop independent dtype(*)
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop independent async
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented}}
#pragma acc loop independent tile(Var, 1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented}}
#pragma acc loop independent gang
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop independent wait
  for(;;);

  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc loop finalize independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc loop if_present independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented}}
#pragma acc loop worker independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented}}
#pragma acc loop vector independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop nohost independent
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(none) independent
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(1) independent
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self independent
  for(;;);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop copy(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop pcopy(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copy(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc loop use_device(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc loop delete(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc loop detach(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc loop device(VarPtr) independent
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(VarPtr) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop device_resident(VarPtr) independent
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop firstprivate(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc loop host(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop link(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop present(Var) independent
  for(;;);
#pragma acc loop private(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop create(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop pcreate(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_create(Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented}}
#pragma acc loop reduction(+:Var) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented}}
#pragma acc loop collapse(1) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop bind(Var) independent
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1) independent
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1) independent
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented}}
#pragma acc loop device_num(1) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented}}
#pragma acc loop default_async(1) independent
  for(;;);
#pragma acc loop device_type(*) independent
  for(;;);
#pragma acc loop dtype(*) independent
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented}}
#pragma acc loop tile(Var, 1) independent
  for(;;);
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented}}
#pragma acc loop gang independent
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait independent
  for(;;);

  // 'seq' cannot be combined with 'gang', 'worker' or 'vector'
  // expected-error@+3{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+2{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented}}
#pragma acc loop seq gang
  for(;;);
  // expected-error@+3{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+2{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented}}
#pragma acc loop seq worker
  for(;;);
  // expected-error@+3{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+2{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented}}
#pragma acc loop seq vector
  for(;;);
  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc loop seq finalize
  for(;;);
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc loop seq if_present
  for(;;);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop seq nohost
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop seq default(none)
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop seq if(1)
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop seq self
  for(;;);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop seq copy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcopy(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_copy(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc loop seq use_device(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop seq attach(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc loop seq delete(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc loop seq detach(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc loop seq device(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop seq deviceptr(VarPtr)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop seq device_resident(VarPtr)
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop seq firstprivate(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc loop seq host(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop seq link(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop seq no_create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop seq present(Var)
  for(;;);
#pragma acc loop seq private(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop seq copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcopyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_copyout(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop seq copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcopyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_copyin(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop seq create(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcreate(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_create(Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented}}
#pragma acc loop seq reduction(+:Var)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented}}
#pragma acc loop seq collapse(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop seq bind(Var)
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop seq vector_length(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop seq num_gangs(1)
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop seq num_workers(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented}}
#pragma acc loop seq device_num(1)
  for(;;);
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented}}
#pragma acc loop seq default_async(1)
  for(;;);
#pragma acc loop seq device_type(*)
  for(;;);
#pragma acc loop seq dtype(*)
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop seq async
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented}}
#pragma acc loop seq tile(Var, 1)
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop seq wait
  for(;;);

  // TODO OpenACC: when 'gang' is implemented and makes it to the AST, this should diagnose because of a conflict with 'seq'.
  // TODOexpected-error@+3{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // TODOexpected-note@+2{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'gang' not yet implemented}}
#pragma acc loop gang seq
  for(;;);
  // TODO OpenACC: when 'worker' is implemented and makes it to the AST, this should diagnose because of a conflict with 'seq'.
  // TODOexpected-error@+3{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // TODOexpected-note@+2{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'worker' not yet implemented}}
#pragma acc loop worker seq
  for(;;);
  // TODO OpenACC: when 'vector' is implemented and makes it to the AST, this should diagnose because of a conflict with 'seq'.
  // TODOexpected-error@+3{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // TODOexpected-note@+2{{previous clause is here}}
  // expected-warning@+1{{OpenACC clause 'vector' not yet implemented}}
#pragma acc loop vector seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc loop finalize seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc loop if_present seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop nohost seq
  for(;;);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(none) seq
  for(;;);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(1) seq
  for(;;);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self seq
  for(;;);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop copy(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop pcopy(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copy(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc loop use_device(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc loop delete(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc loop detach(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device' not yet implemented}}
#pragma acc loop device(VarPtr) seq
  for(;;);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(VarPtr) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop device_resident(VarPtr) seq
  for(;;);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop firstprivate(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'host' not yet implemented}}
#pragma acc loop host(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop link(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop present(Var) seq
  for(;;);
#pragma acc loop private(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop create(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop pcreate(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_create(Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'reduction' not yet implemented}}
#pragma acc loop reduction(+:Var) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'collapse' not yet implemented}}
#pragma acc loop collapse(1) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop bind(Var) seq
  for(;;);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1) seq
  for(;;);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1) seq
  for(;;);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'device_num' not yet implemented}}
#pragma acc loop device_num(1) seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'default_async' not yet implemented}}
#pragma acc loop default_async(1) seq
  for(;;);
#pragma acc loop device_type(*) seq
  for(;;);
#pragma acc loop dtype(*) seq
  for(;;);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async seq
  for(;;);
  // expected-warning@+1{{OpenACC clause 'tile' not yet implemented}}
#pragma acc loop tile(Var, 1) seq
  for(;;);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait seq
  for(;;);
}
