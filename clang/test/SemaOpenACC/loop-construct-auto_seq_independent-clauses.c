// RUN: %clang_cc1 %s -fopenacc -verify

void uses() {
#pragma acc loop auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC clause 'seq' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop auto seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'independent' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop auto independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'auto' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'independent' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'auto' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop independent auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'seq' on 'loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop independent seq
  for(unsigned i = 0; i < 5; ++i);

  int Var;
  int *VarPtr;

  // 'auto' can combine with any other clause.
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'loop' directive}}
#pragma acc loop auto finalize
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'loop' directive}}
#pragma acc loop auto if_present
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto worker
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto vector
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop auto nohost
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop auto default(none)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop auto if(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop auto self
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop auto copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcopy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'loop' directive}}
#pragma acc loop auto use_device(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop auto attach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'loop' directive}}
#pragma acc loop auto delete(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'loop' directive}}
#pragma acc loop auto detach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'loop' directive}}
#pragma acc loop auto device(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop auto deviceptr(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop auto device_resident(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop auto firstprivate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'loop' directive}}
#pragma acc loop auto host(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop auto link(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop auto no_create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop auto present(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto private(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop auto copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcopyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop auto copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcopyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop auto create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop auto pcreate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop auto present_or_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto reduction(+:Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto collapse(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop auto bind(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop auto vector_length(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop auto num_gangs(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop auto num_workers(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'loop' directive}}
#pragma acc loop auto device_num(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'loop' directive}}
#pragma acc loop auto default_async(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto device_type(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto dtype(*)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop auto async
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto tile(1+2, 1)
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc loop auto gang
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop auto wait
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'loop' directive}}
#pragma acc loop finalize auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'loop' directive}}
#pragma acc loop if_present auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop worker auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop vector auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop nohost auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(none) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop copy(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop pcopy(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copy(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'loop' directive}}
#pragma acc loop use_device(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'loop' directive}}
#pragma acc loop delete(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'loop' directive}}
#pragma acc loop detach(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'loop' directive}}
#pragma acc loop device(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop device_resident(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop firstprivate(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'loop' directive}}
#pragma acc loop host(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop link(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop present(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop private(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop create(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop pcreate(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_create(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop reduction(+:Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop collapse(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop bind(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'loop' directive}}
#pragma acc loop device_num(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'loop' directive}}
#pragma acc loop default_async(1) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop device_type(*) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop dtype(*) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop tile(1+2, 1) auto
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc loop gang auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait auto
  for(unsigned i = 0; i < 5; ++i);

  // 'independent' can also be combined with any clauses
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'loop' directive}}
#pragma acc loop independent finalize
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'loop' directive}}
#pragma acc loop independent if_present
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent worker
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent vector
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop independent nohost
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop independent default(none)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop independent if(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop independent self
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop independent copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcopy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'loop' directive}}
#pragma acc loop independent use_device(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop independent attach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'loop' directive}}
#pragma acc loop independent delete(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'loop' directive}}
#pragma acc loop independent detach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'loop' directive}}
#pragma acc loop independent device(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop independent deviceptr(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop independent device_resident(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop independent firstprivate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'loop' directive}}
#pragma acc loop independent host(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop independent link(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop independent no_create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop independent present(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent private(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop independent copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcopyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop independent copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcopyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop independent create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop independent pcreate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop independent present_or_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent reduction(+:Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent collapse(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop independent bind(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop independent vector_length(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop independent num_gangs(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop independent num_workers(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'loop' directive}}
#pragma acc loop independent device_num(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'loop' directive}}
#pragma acc loop independent default_async(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent device_type(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent dtype(*)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop independent async
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent tile(1+2, 1)
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc loop independent gang
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop independent wait
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'loop' directive}}
#pragma acc loop finalize independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'loop' directive}}
#pragma acc loop if_present independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop worker independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop vector independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop nohost independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(none) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop copy(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop pcopy(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copy(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'loop' directive}}
#pragma acc loop use_device(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'loop' directive}}
#pragma acc loop delete(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'loop' directive}}
#pragma acc loop detach(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'loop' directive}}
#pragma acc loop device(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop device_resident(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop firstprivate(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'loop' directive}}
#pragma acc loop host(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop link(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop present(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop private(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop create(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop pcreate(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_create(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop reduction(+:Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop collapse(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop bind(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'loop' directive}}
#pragma acc loop device_num(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'loop' directive}}
#pragma acc loop default_async(1) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop device_type(*) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop dtype(*) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop tile(1+2, 1) independent
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc loop gang independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait independent
  for(unsigned i = 0; i < 5; ++i);

  // 'seq' cannot be combined with 'gang', 'worker' or 'vector'
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq gang
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq worker
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop seq vector
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'loop' directive}}
#pragma acc loop seq finalize
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'loop' directive}}
#pragma acc loop seq if_present
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop seq nohost
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop seq default(none)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop seq if(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop seq self
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop seq copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcopy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'loop' directive}}
#pragma acc loop seq use_device(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop seq attach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'loop' directive}}
#pragma acc loop seq delete(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'loop' directive}}
#pragma acc loop seq detach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'loop' directive}}
#pragma acc loop seq device(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop seq deviceptr(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop seq device_resident(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop seq firstprivate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'loop' directive}}
#pragma acc loop seq host(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop seq link(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop seq no_create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop seq present(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq private(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop seq copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcopyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop seq copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcopyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop seq create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop seq pcreate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop seq present_or_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq reduction(+:Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq collapse(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop seq bind(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop seq vector_length(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop seq num_gangs(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop seq num_workers(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'loop' directive}}
#pragma acc loop seq device_num(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'loop' directive}}
#pragma acc loop seq default_async(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq device_type(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq dtype(*)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop seq async
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop seq tile(1+2, 1)
  for(;;)
    for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop seq wait
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'gang' clause on a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop gang seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop worker seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'vector' clause on a 'loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop vector seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'loop' directive}}
#pragma acc loop finalize seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'loop' directive}}
#pragma acc loop if_present seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc loop nohost seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(none) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if' clause is not valid on 'loop' directive}}
#pragma acc loop if(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'self' clause is not valid on 'loop' directive}}
#pragma acc loop self seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'loop' directive}}
#pragma acc loop copy(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopy' clause is not valid on 'loop' directive}}
#pragma acc loop pcopy(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copy' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copy(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'loop' directive}}
#pragma acc loop use_device(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'attach' clause is not valid on 'loop' directive}}
#pragma acc loop attach(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'loop' directive}}
#pragma acc loop delete(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'loop' directive}}
#pragma acc loop detach(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'loop' directive}}
#pragma acc loop device(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'deviceptr' clause is not valid on 'loop' directive}}
#pragma acc loop deviceptr(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc loop device_resident(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'firstprivate' clause is not valid on 'loop' directive}}
#pragma acc loop firstprivate(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'loop' directive}}
#pragma acc loop host(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc loop link(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'no_create' clause is not valid on 'loop' directive}}
#pragma acc loop no_create(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present' clause is not valid on 'loop' directive}}
#pragma acc loop present(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop private(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyout' clause is not valid on 'loop' directive}}
#pragma acc loop copyout(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyout' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyout(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyout' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyout(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'copyin' clause is not valid on 'loop' directive}}
#pragma acc loop copyin(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcopyin' clause is not valid on 'loop' directive}}
#pragma acc loop pcopyin(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_copyin' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_copyin(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'create' clause is not valid on 'loop' directive}}
#pragma acc loop create(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'pcreate' clause is not valid on 'loop' directive}}
#pragma acc loop pcreate(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'present_or_create' clause is not valid on 'loop' directive}}
#pragma acc loop present_or_create(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop reduction(+:Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop collapse(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc loop bind(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'vector_length' clause is not valid on 'loop' directive}}
#pragma acc loop vector_length(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'loop' directive}}
#pragma acc loop num_gangs(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'num_workers' clause is not valid on 'loop' directive}}
#pragma acc loop num_workers(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'loop' directive}}
#pragma acc loop device_num(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'loop' directive}}
#pragma acc loop default_async(1) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop device_type(*) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop dtype(*) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'loop' directive}}
#pragma acc loop async seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc loop tile(1+2, 1) seq
  for(;;)
    for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait seq
  for(unsigned i = 0; i < 5; ++i);
}
