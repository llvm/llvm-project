// RUN: %clang_cc1 %s -fopenacc -verify -Wno-empty-body -Wno-unused-value

void HasStmt() {
  {
    // expected-error@+2{{expected statement}}
#pragma acc data
  }
  {
    // expected-error@+2{{expected statement}}
#pragma acc host_data
  }
  // Don't have statements, so this is fine.
  {
#pragma acc enter data
  }
  {
#pragma acc exit data
  }
}

void AtLeastOneOf() {
  int Var;
// Data
#pragma acc data copy(Var)
  ;
  // expected-warning@+1{{OpenACC clause 'copyin' not yet implemented}}
#pragma acc data copyin(Var)
  ;
  // expected-warning@+1{{OpenACC clause 'copyout' not yet implemented}}
#pragma acc data copyout(Var)
  ;
#pragma acc data create(Var)
  ;
#pragma acc data no_create(Var)
  ;
  // expected-warning@+1{{OpenACC clause 'present' not yet implemented}}
#pragma acc data present(Var)
  ;
  // expected-warning@+1{{OpenACC clause 'deviceptr' not yet implemented}}
#pragma acc data deviceptr(Var)
  ;
  // expected-warning@+1{{OpenACC clause 'attach' not yet implemented}}
#pragma acc data attach(Var)
  ;
#pragma acc data default(none)
  ;

  // OpenACC TODO: The following 'data' directives should diagnose, since they
  // don't have at least one of the above clauses.

#pragma acc data if(Var)
  ;

#pragma acc data async
  ;

#pragma acc data wait
  ;

#pragma acc data device_type(*)
  ;
#pragma acc data
  ;

  // Enter Data
  // expected-warning@+1{{OpenACC clause 'copyin' not yet implemented}}
#pragma acc enter data copyin(Var)
#pragma acc enter data create(Var)
  // expected-warning@+1{{OpenACC clause 'attach' not yet implemented}}
#pragma acc enter data attach(Var)

  // OpenACC TODO: The following 'enter data' directives should diagnose, since
  // they don't have at least one of the above clauses.

#pragma acc enter data if(Var)
#pragma acc enter data async
#pragma acc enter data wait
#pragma acc enter data

  // Exit Data
  // expected-warning@+1{{OpenACC clause 'copyout' not yet implemented}}
#pragma acc exit data copyout(Var)
  // expected-warning@+1{{OpenACC clause 'delete' not yet implemented}}
#pragma acc exit data delete(Var)
  // expected-warning@+1{{OpenACC clause 'detach' not yet implemented}}
#pragma acc exit data detach(Var)

  // OpenACC TODO: The following 'exit data' directives should diagnose, since
  // they don't have at least one of the above clauses.

#pragma acc exit data if(Var)
#pragma acc exit data async
#pragma acc exit data wait
  // expected-warning@+1{{OpenACC clause 'finalize' not yet implemented}}
#pragma acc exit data finalize
#pragma acc exit data

  // Host Data
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Var)
  ;
  // OpenACC TODO: The following 'host_data' directives should diagnose, since
  // they don't have at least one of the above clauses.

#pragma acc host_data if(Var)
  ;
  // expected-warning@+1{{OpenACC clause 'if_present' not yet implemented}}
#pragma acc host_data if_present
  ;
#pragma acc host_data
  ;
}

void DataRules() {
  int Var;
  // OpenACC TODO: Only 'async' and 'wait' are permitted after a device_type, so
  // the rest of these should diagnose.

  // expected-error@+2{{OpenACC clause 'copy' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) copy(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyin' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) copyin(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyout' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) copyout(Var)
  ;
  // expected-error@+2{{OpenACC clause 'create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'no_create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) no_create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'present' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) present(Var)
  ;
  // expected-error@+2{{OpenACC clause 'deviceptr' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) deviceptr(Var)
  ;
  // expected-error@+2{{OpenACC clause 'attach' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) attach(Var)
  ;
  // expected-error@+2{{OpenACC clause 'default' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) default(none)
  ;
  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(*) if(Var)
  ;
#pragma acc data device_type(*) async
  ;
#pragma acc data device_type(*) wait
  ;
}

struct HasMembers {
  int Member;

  void HostDataError() {
  // TODO OpenACC: The following 3 should error, as use_device's var only allows
  // a variable or array, not an array index, or sub expression.

  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(this)
  ;
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(this->Member)
  ;
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Member)
  ;
  }
};

void HostDataRules() {
  int Var, Var2;
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'host_data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc host_data if(Var) if (Var2)
  ;

  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Var)
  ;

  int Array[5];
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Array)
  ;

  // TODO OpenACC: The following 3 should error, as use_device's var only allows
  // a variable or array, not an array index, or sub expression.

  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Array[1:1])
  ;

  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Array[1])
  ;
  HasMembers HM;
  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(HM.Member)
  ;

}
