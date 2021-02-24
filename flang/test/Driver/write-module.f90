! RUN: rm -rf %t && mkdir -p %t/mod-dir && cd %t && %f18 -fparse-only %s
! RUN: ls %t/testmodule.mod && not ls %t/mod-dir/testmodule.mod

! RUN: rm -rf %t && mkdir -p %t/mod-dir && cd %t && %f18 -fparse-only -module mod-dir %s
! RUN: ls %t/mod-dir/testmodule.mod && not ls %t/testmodule.mod

! RUN: rm -rf %t && mkdir -p %t/mod-dir && cd %t && %f18 -fparse-only -module-dir mod-dir %s
! RUN: ls %t/mod-dir/testmodule.mod && not ls %t/testmodule.mod

! RUN: rm -rf %t && mkdir -p %t/mod-dir && cd %t && %f18 -fparse-only -J mod-dir %s
! RUN: ls %t/mod-dir/testmodule.mod && not ls %t/testmodule.mod

! RUN: rm -rf %t && mkdir -p %t/mod-dir && cd %t && %f18 -fparse-only -Jmod-dir %s
! RUN: ls %t/mod-dir/testmodule.mod && not ls %t/testmodule.mod

module testmodule
  type::t2
  end type
end
