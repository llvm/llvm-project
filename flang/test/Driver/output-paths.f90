! Test the diagnostic for cases when the output file cannot be generated

!--------------------------
! RUN lines
!--------------------------
! RUN: not %flang_fc1 -E -o %t.doesnotexist/somename %s 2> %t
! RUN: FileCheck -check-prefix=OUTPUTFAIL -DMSG=%errc_ENOENT -input-file=%t %s

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! OUTPUTFAIL: error: unable to open output file '{{.*}}doesnotexist{{.}}somename': '[[MSG]]'
