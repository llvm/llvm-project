// RUN: %clang_cc1 %s -fopenacc -verify -Wno-empty-body -Wno-unused-value

void HasStmt() {
  {
    // expected-error@+2{{expected statement}}
#pragma acc data default(none)
  }

  int I;
  {
    // expected-error@+2{{expected statement}}
#pragma acc host_data use_device(I)
  }
  // Don't have statements, so this is fine.
  {
#pragma acc enter data copyin(I)
  }
  {
#pragma acc exit data copyout(I)
  }
}

void AtLeastOneOf() {
  int Var;
  int *VarPtr = &Var;
// Data
#pragma acc data copy(Var)
  ;
#pragma acc data copyin(Var)
  ;
#pragma acc data copyout(Var)
  ;
#pragma acc data create(Var)
  ;
#pragma acc data no_create(Var)
  ;
#pragma acc data present(Var)
  ;
#pragma acc data deviceptr(VarPtr)
  ;
#pragma acc data attach(VarPtr)
  ;
#pragma acc data default(none)
  ;

    // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data if(Var)
  ;

    // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data async
  ;

    // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data wait
  ;

    // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data device_type(*)
  ;
    // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data
  ;

  // Enter Data
#pragma acc enter data copyin(Var)
#pragma acc enter data create(Var)
#pragma acc enter data attach(VarPtr)

  // expected-error@+1{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
#pragma acc enter data if(Var)
  // expected-error@+1{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
#pragma acc enter data async
  // expected-error@+1{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
#pragma acc enter data wait
  // expected-error@+1{{OpenACC 'enter data' construct must have at least one 'copyin', 'create' or 'attach' clause}}
#pragma acc enter data

  // Exit Data
#pragma acc exit data copyout(Var)
#pragma acc exit data delete(Var)
#pragma acc exit data detach(VarPtr)

  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
#pragma acc exit data if(Var)
  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
#pragma acc exit data async
  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
#pragma acc exit data wait
  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
#pragma acc exit data finalize
  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete' or 'detach' clause}}
#pragma acc exit data

  // Host Data
#pragma acc host_data use_device(Var)
  ;

  // expected-error@+1{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
#pragma acc host_data if(Var)
  ;
  // expected-error@+1{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
#pragma acc host_data if_present
  ;
  // expected-error@+1{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
#pragma acc host_data
  ;
}

void DataRules() {
  int Var;
  // expected-error@+2{{OpenACC clause 'copy' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) copy(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyin' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) copyin(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyout' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) copyout(Var)
  ;
  // expected-error@+2{{OpenACC clause 'create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'no_create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) no_create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'present' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) present(Var)
  ;
  // expected-error@+2{{OpenACC clause 'deviceptr' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) deviceptr(Var)
  ;
  // expected-error@+2{{OpenACC clause 'attach' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) attach(Var)
  ;
  // expected-error@+2{{OpenACC clause 'default' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) default(none)
  ;
  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) device_type(*) if(Var)
  ;
#pragma acc data default(none) device_type(*) async
  ;
#pragma acc data default(none) device_type(*) wait
  ;
}

struct HasMembers {
  int Member;

  void HostDataError() {
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(this)
  ;
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(this->Member)
  ;
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(Member)
  ;
  }
};

void HostDataRules() {
  int Var, Var2;
  // expected-error@+3{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'host_data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc host_data if(Var) if (Var2)
  ;

#pragma acc host_data use_device(Var)
  ;

  int Array[5];
#pragma acc host_data use_device(Array)
  ;

  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(Array[1:1])
  ;

  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(Array[1])
  ;
  HasMembers HM;
  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(HM.Member)
  ;

}
