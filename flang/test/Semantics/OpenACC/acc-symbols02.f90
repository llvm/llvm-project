! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenacc

!DEF:/acc_declare_symbolsModule
module acc_declare_symbols
  !DEF: /acc_declare_symbols/a PUBLIC (AccCreate, AccDeclare) ObjectEntity REAL(4)
  real a(100)
  !$acc declare create(a)

  !DEF:/acc_declare_symbols/b PUBLIC (AccCopyIn, AccDeclare) ObjectEntity REAL(4)
  real b(20)
  !$acc declare copyin(b)

  !DEF:/acc_declare_symbols/c PUBLIC (AccDeviceResident, AccDeclare) ObjectEntity REAL(4)
  real c(10)
  !$acc declare device_resident(c)

  !DEF:/acc_declare_symbols/d PUBLIC (AccLink, AccDeclare) ObjectEntity REAL(4)
  real d(10)
  !$acc declare link(d)

end module
