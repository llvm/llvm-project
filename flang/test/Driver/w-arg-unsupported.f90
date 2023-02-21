! RUN: %flang -std=f2018 -Wextra -Waliasing -Wampersand -Warray-bounds -Wc-binding-type \
! RUN:        -Wcharacter-truncation -Wconversion -Wdo-subscript -Wfunction-elimination \
! RUN:        -Wimplicit-interface -Wimplicit-procedure -Wintrinsic-shadow -Wuse-without-only \
! RUN:        -Wintrinsics-std -Wline-truncation -Wno-align-commons -Wno-overwrite-recursive \
! RUN:        -Wno-tabs -Wreal-q-constant -Wsurprising -Wunderflow -Wunused-parameter \
! RUN:        -Wrealloc-lhs -Wrealloc-lhs-all -Wfrontend-loop-interchange -Wtarget-lifetime %s \
! RUN:        2>&1 | FileCheck %s

! CHECK: The warning option '-Wextra' is not supported
! CHECK-NEXT: The warning option '-Waliasing' is not supported
! CHECK-NEXT: The warning option '-Wampersand' is not supported
! CHECK-NEXT: The warning option '-Warray-bounds' is not supported
! CHECK-NEXT: The warning option '-Wc-binding-type' is not supported
! CHECK-NEXT: The warning option '-Wcharacter-truncation' is not supported
! CHECK-NEXT: The warning option '-Wconversion' is not supported
! CHECK-NEXT: The warning option '-Wdo-subscript' is not supported
! CHECK-NEXT: The warning option '-Wfunction-elimination' is not supported
! CHECK-NEXT: The warning option '-Wimplicit-interface' is not supported
! CHECK-NEXT: The warning option '-Wimplicit-procedure' is not supported
! CHECK-NEXT: The warning option '-Wintrinsic-shadow' is not supported
! CHECK-NEXT: The warning option '-Wuse-without-only' is not supported
! CHECK-NEXT: The warning option '-Wintrinsics-std' is not supported
! CHECK-NEXT: The warning option '-Wline-truncation' is not supported
! CHECK-NEXT: The warning option '-Wno-align-commons' is not supported
! CHECK-NEXT: The warning option '-Wno-overwrite-recursive' is not supported
! CHECK-NEXT: The warning option '-Wno-tabs' is not supported
! CHECK-NEXT: The warning option '-Wreal-q-constant' is not supported
! CHECK-NEXT: The warning option '-Wsurprising' is not supported
! CHECK-NEXT: The warning option '-Wunderflow' is not supported
! CHECK-NEXT: The warning option '-Wunused-parameter' is not supported
! CHECK-NEXT: The warning option '-Wrealloc-lhs' is not supported
! CHECK-NEXT: The warning option '-Wrealloc-lhs-all' is not supported
! CHECK-NEXT: The warning option '-Wfrontend-loop-interchange' is not supported
! CHECK-NEXT: The warning option '-Wtarget-lifetime' is not supported

program m
end program
