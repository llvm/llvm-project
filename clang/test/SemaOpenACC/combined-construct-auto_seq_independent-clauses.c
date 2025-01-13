// RUN: %clang_cc1 %s -fopenacc -verify

void uses() {
#pragma acc parallel loop auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC clause 'seq' on 'parallel loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop auto seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'independent' on 'parallel loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop auto independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'auto' on 'parallel loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop seq auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'independent' on 'parallel loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop seq independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'auto' on 'parallel loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop independent auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'seq' on 'parallel loop' construct conflicts with previous data dependence clause}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop independent seq
  for(unsigned i = 0; i < 5; ++i);

  int Var;
  int *VarPtr;

  // 'auto' can combine with any other clause.
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto finalize
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto if_present
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto worker
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto vector
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc parallel loop auto nohost
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto default(none)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto if(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto self
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop auto pcopy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop auto present_or_copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto use_device(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto attach(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto delete(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto detach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto device(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto deviceptr(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc parallel loop auto device_resident(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto firstprivate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto host(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc parallel loop auto link(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto no_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto present(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto private(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop auto pcopyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop auto present_or_copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop auto pcopyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop auto present_or_copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop auto pcreate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop auto present_or_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto reduction(+:Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto collapse(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc parallel loop auto bind(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto vector_length(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto num_gangs(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto num_workers(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto device_num(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop auto default_async(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto device_type(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto dtype(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto async
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto tile(1+2, 1)
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto gang
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop auto wait
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop finalize auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop if_present auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop worker auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop vector auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc parallel loop nohost auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop default(none) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop if(1) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop self auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copy(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop pcopy(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop present_or_copy(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop use_device(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop attach(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop delete(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop detach(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop deviceptr(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc parallel loop device_resident(VarPtr) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop firstprivate(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop host(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc parallel loop link(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop no_create(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop present(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop private(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copyout(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop pcopyout(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop present_or_copyout(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copyin(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop pcopyin(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop present_or_copyin(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop create(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop pcreate(Var) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop present_or_create(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop reduction(+:Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop collapse(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc parallel loop bind(Var) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop vector_length(1) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop num_gangs(1) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop num_workers(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device_num(1) auto
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop default_async(1) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop dtype(*) auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop async auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop tile(1+2, 1) auto
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop gang auto
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop wait auto
  for(unsigned i = 0; i < 5; ++i);

  // 'independent' can also be combined with any clauses
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent finalize
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent if_present
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent worker
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent vector
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc parallel loop independent nohost
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent default(none)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent if(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent self
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop independent pcopy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop independent present_or_copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent use_device(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent attach(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent delete(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent detach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent device(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent deviceptr(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc parallel loop independent device_resident(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent firstprivate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent host(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc parallel loop independent link(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent no_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent present(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent private(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop independent pcopyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop independent present_or_copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop independent pcopyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop independent present_or_copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop independent pcreate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop independent present_or_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent reduction(+:Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent collapse(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc parallel loop independent bind(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent vector_length(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent num_gangs(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent num_workers(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent device_num(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop independent default_async(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent device_type(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent dtype(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent async
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent tile(1+2, 1)
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent gang
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop independent wait
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop finalize independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop if_present independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop worker independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop vector independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc parallel loop nohost independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop default(none) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop if(1) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop self independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copy(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop pcopy(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop present_or_copy(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop use_device(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop attach(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop delete(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop detach(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop deviceptr(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc parallel loop device_resident(VarPtr) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop firstprivate(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop host(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc parallel loop link(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop no_create(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop present(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop private(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copyout(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop pcopyout(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop present_or_copyout(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copyin(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop pcopyin(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop present_or_copyin(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop create(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop pcreate(Var) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop present_or_create(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop reduction(+:Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop collapse(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc parallel loop bind(Var) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop vector_length(1) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop num_gangs(1) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop num_workers(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device_num(1) independent
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop default_async(1) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop dtype(*) independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop async independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop tile(1+2, 1) independent
  for(unsigned j = 0; j < 5; ++j)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop gang independent
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop wait independent
  for(unsigned i = 0; i < 5; ++i);

  // 'seq' cannot be combined with 'gang', 'worker' or 'vector'
  // expected-error@+2{{OpenACC clause 'gang' may not appear on the same construct as a 'seq' clause on a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop seq gang
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'worker' may not appear on the same construct as a 'seq' clause on a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop seq worker
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'seq' clause on a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop seq vector
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq finalize
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq if_present
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc parallel loop seq nohost
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq default(none)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq if(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq self
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop seq pcopy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop seq present_or_copy(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq use_device(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq attach(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq delete(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq detach(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq device(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq deviceptr(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc parallel loop seq device_resident(VarPtr)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq firstprivate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq host(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc parallel loop seq link(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq no_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq present(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq private(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop seq pcopyout(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop seq present_or_copyout(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop seq pcopyin(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop seq present_or_copyin(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq create(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop seq pcreate(Var)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop seq present_or_create(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq reduction(+:Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq collapse(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc parallel loop seq bind(Var)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq vector_length(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq num_gangs(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq num_workers(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq device_num(1)
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop seq default_async(1)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq device_type(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq dtype(*)
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq async
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq tile(1+2, 1)
  for(;;)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop seq wait
  for(unsigned i = 0; i < 5; ++i);

  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'gang' clause on a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop gang seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop worker seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+2{{OpenACC clause 'seq' may not appear on the same construct as a 'vector' clause on a 'parallel loop' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc parallel loop vector seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'finalize' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop finalize seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'if_present' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop if_present seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'nohost' not yet implemented}}
#pragma acc parallel loop nohost seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop default(none) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop if(1) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop self seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copy(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop pcopy(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc parallel loop present_or_copy(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'use_device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop use_device(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop attach(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'delete' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop delete(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'detach' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop detach(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop deviceptr(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'device_resident' not yet implemented}}
#pragma acc parallel loop device_resident(VarPtr) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop firstprivate(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'host' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop host(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'link' not yet implemented}}
#pragma acc parallel loop link(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop no_create(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop present(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop private(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copyout(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop pcopyout(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc parallel loop present_or_copyout(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop copyin(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop pcopyin(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc parallel loop present_or_copyin(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop create(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop pcreate(Var) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc parallel loop present_or_create(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop reduction(+:Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop collapse(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-warning@+1{{OpenACC clause 'bind' not yet implemented}}
#pragma acc parallel loop bind(Var) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop vector_length(1) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop num_gangs(1) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop num_workers(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'device_num' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop device_num(1) seq
  for(unsigned i = 0; i < 5; ++i);
  // expected-error@+1{{OpenACC 'default_async' clause is not valid on 'parallel loop' directive}}
#pragma acc parallel loop default_async(1) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(*) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop dtype(*) seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop async seq
  for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop tile(1+2, 1) seq
  for(;;)
    for(unsigned i = 0; i < 5; ++i);
#pragma acc parallel loop wait seq
  for(unsigned i = 0; i < 5; ++i);
}
