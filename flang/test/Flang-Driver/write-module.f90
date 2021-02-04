! RUN: mkdir -p %t/dir-f18 && %f18 -fparse-only -I tools/flang/include/flang -module %t/dir-f18 %s  2>&1
! RUN: ls %t/dir-f18/testmodule.mod && not ls %t/testmodule.mod

! RUN: mkdir -p %t/dir-flang-new && %flang-new -fsyntax-only -module-dir %t/dir-flang-new %s  2>&1
! RUN: ls %t/dir-flang-new/testmodule.mod && not ls %t/testmodule.mod

module testmodule
  type::t2
  end type
end
