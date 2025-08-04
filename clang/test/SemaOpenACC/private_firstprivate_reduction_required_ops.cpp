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

struct DeletedCopy {
  DeletedCopy(const DeletedCopy&) = delete;
};

struct DefaultedCopy {
  DefaultedCopy(const DefaultedCopy&) = default;
};
struct UserCopy {
  UserCopy(const UserCopy&);
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

void firstprivate_uses(ImplicitCtorDtor &CDT, ImplDeletedCtor &IDC,
                  DefaultedCtor &DefC, ImpledCtor &IC, DeletedCtor &DelC,
                  ImpledDtor &ID, DefaultedDtor &DefD, DeletedDtor &DelD,
                  ImplicitDelDtor &IDD, DeletedCopy &DelCopy,
                  DefaultedCopy &DefCopy, UserCopy &UDCopy) {
#pragma acc parallel firstprivate(CDT)
  ;

#pragma acc parallel firstprivate(IDC)
  ;

#pragma acc parallel firstprivate(DefC)
  ;

#pragma acc parallel firstprivate(IC)
  ;

#pragma acc parallel firstprivate(DelC)
  ;

#pragma acc parallel firstprivate(ID)
  ;

#pragma acc parallel firstprivate(DefD)
  ;

  // expected-error@+1{{variable of type 'DeletedDtor' referenced in OpenACC 'firstprivate' clause does not have a destructor; reference has no effect}}
#pragma acc parallel firstprivate(DelD)
  ;

  // expected-error@+1{{variable of type 'ImplicitDelDtor' referenced in OpenACC 'firstprivate' clause does not have a copy constructor; reference has no effect}}
#pragma acc parallel firstprivate(IDD)
  ;

  // expected-error@+1{{variable of type 'DeletedCopy' referenced in OpenACC 'firstprivate' clause does not have a copy constructor; reference has no effect}}
#pragma acc parallel firstprivate(DelCopy)
  ;
#pragma acc parallel firstprivate(DefCopy)
  ;
#pragma acc parallel firstprivate(UDCopy)
  ;
}

template<typename T>
void firstprivate_template(T& t) {
#pragma acc parallel firstprivate(t) // #FIRSTPRIV
  ;
}

void firstprivate_inst(ImplicitCtorDtor &CDT, ImplDeletedCtor &IDC,
                       DefaultedCtor &DefC, ImpledCtor &IC, DeletedCtor &DelC,
                       ImpledDtor &ID, DefaultedDtor &DefD, DeletedDtor &DelD,
                       ImplicitDelDtor &IDD, DeletedCopy &DelCopy,
                       DefaultedCopy &DefCopy, UserCopy &UDCopy) {
  firstprivate_template(CDT);
  firstprivate_template(IDC);
  firstprivate_template(DefC);
  firstprivate_template(IC);
  firstprivate_template(DelC);
  firstprivate_template(ID);
  firstprivate_template(DefD);
  // expected-error@#FIRSTPRIV{{variable of type 'DeletedDtor' referenced in OpenACC 'firstprivate' clause does not have a destructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  firstprivate_template(DelD);
  // expected-error@#FIRSTPRIV{{variable of type 'ImplicitDelDtor' referenced in OpenACC 'firstprivate' clause does not have a copy constructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  firstprivate_template(IDD);
  // expected-error@#FIRSTPRIV{{variable of type 'DeletedCopy' referenced in OpenACC 'firstprivate' clause does not have a copy constructor; reference has no effect}}
  // expected-note@+1{{in instantiation}}
  firstprivate_template(DelCopy);
  firstprivate_template(DefCopy);
  firstprivate_template(UDCopy);
}

