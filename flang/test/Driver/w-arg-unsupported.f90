! RUN: %flang -std=f2018 -Wextra -Waliasing -Wampersand -Warray-bounds -Wc-binding-type \
! RUN:        -Wcharacter-truncation -Wconversion -Wdo-subscript -Wfunction-elimination \
! RUN:        -Wimplicit-interface -Wimplicit-procedure -Wintrinsic-shadow -Wuse-without-only \
! RUN:        -Wintrinsics-std -Wline-truncation -Wno-align-commons -Wno-overwrite-recursive \
! RUN:        -Wno-tabs -Wreal-q-constant -Wsurprising -Wunderflow -Wunused-parameter \
! RUN:        -Wrealloc-lhs -Wrealloc-lhs-all -Wfrontend-loop-interchange -Wtarget-lifetime %s \
! RUN:        -c 2>&1 | FileCheck %s

! CHECK: the warning option '-Wextra' is not supported
! CHECK-NEXT: the warning option '-Waliasing' is not supported
! CHECK-NEXT: the warning option '-Wampersand' is not supported
! CHECK-NEXT: the warning option '-Warray-bounds' is not supported
! CHECK-NEXT: the warning option '-Wc-binding-type' is not supported
! CHECK-NEXT: the warning option '-Wcharacter-truncation' is not supported
! CHECK-NEXT: the warning option '-Wconversion' is not supported
! CHECK-NEXT: the warning option '-Wdo-subscript' is not supported
! CHECK-NEXT: the warning option '-Wfunction-elimination' is not supported
! CHECK-NEXT: the warning option '-Wimplicit-interface' is not supported
! CHECK-NEXT: the warning option '-Wimplicit-procedure' is not supported
! CHECK-NEXT: the warning option '-Wintrinsic-shadow' is not supported
! CHECK-NEXT: the warning option '-Wuse-without-only' is not supported
! CHECK-NEXT: the warning option '-Wintrinsics-std' is not supported
! CHECK-NEXT: the warning option '-Wline-truncation' is not supported
! CHECK-NEXT: the warning option '-Wno-align-commons' is not supported
! CHECK-NEXT: the warning option '-Wno-overwrite-recursive' is not supported
! CHECK-NEXT: the warning option '-Wno-tabs' is not supported
! CHECK-NEXT: the warning option '-Wreal-q-constant' is not supported
! CHECK-NEXT: the warning option '-Wsurprising' is not supported
! CHECK-NEXT: the warning option '-Wunderflow' is not supported
! CHECK-NEXT: the warning option '-Wunused-parameter' is not supported
! CHECK-NEXT: the warning option '-Wrealloc-lhs' is not supported
! CHECK-NEXT: the warning option '-Wrealloc-lhs-all' is not supported
! CHECK-NEXT: the warning option '-Wfrontend-loop-interchange' is not supported
! CHECK-NEXT: the warning option '-Wtarget-lifetime' is not supported

program m
end program
