// RUN: %clang_cc1 %s -fopenacc -verify

struct ImplicitCtorDtor{};

struct ImplDeletedCtor{
  ImplDeletedCtor(int i);
};

struct DefaultedCtor {
  DefaultedCtor() = default;
};

struct ImpledCtor {
  ImpledCtor() = default;
};


struct DeletedCtor {
  DeletedCtor() = delete;
};

struct ImpledDtor {
  ~ImpledDtor();
};

struct DefaultedDtor {
  ~DefaultedDtor() = default;
};

struct DeletedDtor {
  ~DeletedDtor() = delete;
};

struct ImplicitDelDtor {
  DeletedDtor d;
};

void private_uses(ImplicitCtorDtor &CDT, ImplDeletedCtor &IDC,
                  DefaultedCtor &DefC, ImpledCtor &IC, DeletedCtor &DelC,
                  ImpledDtor &ID, DefaultedDtor &DefD, DeletedDtor &DelD,
                  ImplicitDelDtor &IDD) {

#pragma acc parallel private(CDT)
  ;

  // expected-error@+1{{variable of type 'ImplDeletedCtor' referenced in OpenACC 'private' clause does not have a default constructor; reference has no effect}}
#pragma acc parallel private(IDC)
  ;

#pragma acc parallel private(DefC)
  ;

#pragma acc parallel private(IC)
  ;

  // expected-error@+1{{variable of type 'DeletedCtor' referenced in OpenACC 'private' clause does not have a default constructor; reference has no effect}}
#pragma acc parallel private(DelC)
  ;

#pragma acc parallel private(ID)
  ;

#pragma acc parallel private(DefD)
  ;

  // expected-error@+1{{variable of type 'DeletedDtor' referenced in OpenACC 'private' clause does not have a destructor; reference has no effect}}
#pragma acc parallel private(DelD)
  ;

  // expected-error@+1{{variable of type 'ImplicitDelDtor' referenced in OpenACC 'private' clause does not have a destructor; reference has no effect}}
#pragma acc parallel private(IDD)
  ;

}

template<typename T>
void private_templ(T& t) {
#pragma acc parallel private(t) // #PRIV
  ;
}

void inst(ImplicitCtorDtor &CDT, ImplDeletedCtor &IDC,
                  DefaultedCtor &DefC, ImpledCtor &IC, DeletedCtor &DelC,
                  ImpledDtor &ID, DefaultedDtor &DefD, DeletedDtor &DelD,
                  ImplicitDelDtor &IDD) {
  private_templ(CDT);
  // expected-error@#PRIV{{variable of type 'ImplDeletedCtor' referenced in OpenACC 'private' clause does not have a default constructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  private_templ(IDC);
  private_templ(DefC);
  private_templ(IC);
  // expected-error@#PRIV{{variable of type 'DeletedCtor' referenced in OpenACC 'private' clause does not have a default constructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  private_templ(DelC);
  private_templ(ID);
  private_templ(DefD);
  // expected-error@#PRIV{{variable of type 'DeletedDtor' referenced in OpenACC 'private' clause does not have a destructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  private_templ(DelD);
  // expected-error@#PRIV{{variable of type 'ImplicitDelDtor' referenced in OpenACC 'private' clause does not have a destructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  private_templ(IDD);
}
