! Test to check the working of option "-fprofile-sample-use".

! RUN: %flang -### -target x86_64-unknown-linux %s 2>&1 | FileCheck %s --check-prefix=NO-PROFILE-SAMPLE-USE
! RUN: %flang -### -target x86_64-unknown-linux -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s 2>&1 | FileCheck %s --check-prefix=PROFILE-SAMPLE-USE
! RUN: %flang -### -target x86_64-unknown-linux -fno-profile-sample-use %s 2>&1 | FileCheck %s --check-prefix=NO-PROFILE-SAMPLE-USE

! RUN: %flang -### -target x86_64-unknown-linux -fprofile-sample-use=%S/Inputs/pgo-sample.prof -fno-profile-sample-use %s 2>&1 | FileCheck %s --check-prefix=NO-PROFILE-SAMPLE-USE

! RUN: not %flang -target x86_64-unknown-linux -fsyntax-only -fprofile-sample-use=%t/missing-profile.prof %s 2>&1 | FileCheck %s --check-prefix=PROFILE-SAMPLE-USE-NO-FILE
! RUN: not %flang -target x86_64-unknown-linux -fsyntax-only -fprofile-generate -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s 2>&1 | FileCheck %s --check-prefix=PROFILE-SAMPLE-USE-ERROR

! RUN: not %flang -target powerpc64-ibm-aix -### -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PROFILE-SAMPLE-USE-UNSUPPORTED-AIX

! NO-PROFILE-SAMPLE-USE-NOT: "-fprofile-sample-use"
! PROFILE-SAMPLE-USE: "-fprofile-sample-use={{.*}}/Inputs/pgo-sample.prof"

! PROFILE-SAMPLE-USE-NO-FILE: error: no such file or directory: {{.*}}missing-profile.prof{{.*}}
! PROFILE-SAMPLE-USE-ERROR: error: invalid argument '-fprofile-generate' not allowed with '-fprofile-sample-use={{.*}}'

! PROFILE-SAMPLE-USE-UNSUPPORTED-AIX: error: unsupported option '-fprofile-sample-use=' for target 'powerpc64-ibm-aix'

integer function hot(x)
   integer, intent(in) :: x
   hot = x*2
end function hot

integer function cold(x)
   integer, intent(in) :: x
   cold = x - 10
end function

program test_sample_use
    integer :: i, r
    do i = 1, 100
       r = hot(i)
    end do
 end program test_sample_use
