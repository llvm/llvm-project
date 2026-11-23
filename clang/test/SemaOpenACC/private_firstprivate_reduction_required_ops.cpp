// RUN: %clang_cc1 %s -fopenacc -verify

struct ImplicitCtorDtor{};

struct ImplDeletedCtor{
  ImplDeletedCtor(int i);
};

struct DefaultedCtor {
  DefaultedCtor() = default;
};

struct ImpledCtor {
  ImpledCtor();
};


struct DeletedCtor {
  DeletedCtor() = delete;
  DeletedCtor(int i);
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
  DeletedCopy();
  DeletedCopy(const DeletedCopy&) = delete;
};

struct DefaultedCopy {
  DefaultedCopy();
  DefaultedCopy(const DefaultedCopy&) = default;
};
struct UserCopy {
  UserCopy();
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

  // expected-error@+1{{variable of type 'DeletedDtor' referenced in OpenACC 'private' clause does not have a}}
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

void private_arrays() {
  char *ptr;
  ImplicitCtorDtor CDTArr[5];
  ImplDeletedCtor IDCArr[5]{1,2,3,4,5};
  DefaultedCtor DefCArr[5];
  ImpledCtor ICArr[5];
  DeletedCtor DelCArr[5]{1,2,3,4,5};
  ImpledDtor IDArr[5];
  DefaultedDtor DefDArr[5];
  using DelDArrTy = DeletedDtor[5];
  DelDArrTy &DelDArr = *((DelDArrTy*)ptr);
  using IDDArrTy = ImplicitDelDtor[5];
  IDDArrTy &IDDArr = *((IDDArrTy*)ptr);


#pragma acc parallel private(CDTArr)
  ;
  // expected-error@+1{{variable of type 'ImplDeletedCtor' referenced in OpenACC 'private' clause does not have a default constructor; reference has no effect}}
#pragma acc parallel private(IDCArr)
  ;
#pragma acc parallel private(DefCArr)
  ;
#pragma acc parallel private(ICArr)
  ;
  // expected-error@+1{{variable of type 'DeletedCtor' referenced in OpenACC 'private' clause does not have a default constructor; reference has no effect}}
#pragma acc parallel private(DelCArr)
  ;
#pragma acc parallel private(IDArr)
  ;
#pragma acc parallel private(DefDArr)
  ;
  // expected-error@+1{{variable of type 'DeletedDtor' referenced in OpenACC 'private' clause does not have a destructor; reference has no effect}}
#pragma acc parallel private(DelDArr)
  ;
  // expected-error@+1{{variable of type 'ImplicitDelDtor' referenced in OpenACC 'private' clause does not have a destructor; reference has no effect}}
#pragma acc parallel private(IDDArr)
  ;
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

void firstprivate_arrays() {
  char *ptr;
  ImplicitCtorDtor CDTArr[5];
  ImplDeletedCtor IDCArr[5]{1,2,3,4,5};
  DefaultedCtor DefCArr[5];
  ImpledCtor ICArr[5];
  DeletedCtor DelCArr[5]{1,2,3,4,5};
  ImpledDtor IDArr[5];
  DefaultedDtor DefDArr[5];
  using DelDArrTy = DeletedDtor[5];
  DelDArrTy &DelDArr = *((DelDArrTy*)ptr);
  using IDDArrTy = ImplicitDelDtor[5];
  IDDArrTy &IDDArr = *((IDDArrTy*)ptr);
  DeletedCopy DelCopyArr[5]{};
  DefaultedCopy DefCopyArr[5]{};
  UserCopy UDCopyArr[5]{};

#pragma acc parallel firstprivate(CDTArr)
  ;
#pragma acc parallel firstprivate(IDCArr)
  ;
#pragma acc parallel firstprivate(DefCArr)
  ;
#pragma acc parallel firstprivate(ICArr)
  ;
#pragma acc parallel firstprivate(DelCArr)
  ;
#pragma acc parallel firstprivate(IDArr)
  ;
#pragma acc parallel firstprivate(DefDArr)
  ;
  // expected-error@+1{{variable of type 'DeletedDtor' referenced in OpenACC 'firstprivate' clause does not have a destructor; reference has no effect}}
#pragma acc parallel firstprivate(DelDArr)
  ;
  // expected-error@+1{{variable of type 'ImplicitDelDtor' referenced in OpenACC 'firstprivate' clause does not have a copy constructor; reference has no effect}}
#pragma acc parallel firstprivate(IDDArr)
  ;
  // expected-error@+1{{variable of type 'DeletedCopy' referenced in OpenACC 'firstprivate' clause does not have a copy constructor; reference has no effect}}
#pragma acc parallel firstprivate(DelCopyArr)
  ;
#pragma acc parallel firstprivate(DefCopyArr)
  ;
#pragma acc parallel firstprivate(UDCopyArr)
  ;
}

template<unsigned I>
void non_const_array_templ() {
  int CArr[I];

#pragma acc parallel firstprivate(CArr)
  ;
}

void non_const_arrays(int I) {
  non_const_array_templ<5>();

  int NCArr[I];
  // expected-warning@+1{{variable of array type 'int[I]' referenced in OpenACC 'firstprivate' clause does not have constant bounds; initialization will happen after decay to pointer}}
#pragma acc parallel firstprivate(NCArr)
  ;
}
