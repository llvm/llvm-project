! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in sync images statements

program test_sync_images
  implicit none

  integer, parameter :: invalid_rank(*,*) = reshape([1], [1,1])
  integer sync_status, non_scalar(2), superfluous_stat, coindexed_integer[*], me
  character(len=128) error_message, superfluous_errmsg, coindexed_character[*]
  logical invalid_type
  
  !___ standard-conforming statement ___

  sync images(*, stat=sync_status, errmsg=error_message)
  sync images(*, stat=sync_status                      )
  sync images(*,                   errmsg=error_message)
  sync images(*                                        )

  sync images(me,   stat=sync_status, errmsg=error_message)
  sync images(me+1, stat=sync_status, errmsg=error_message)
  sync images(1,    stat=sync_status, errmsg=error_message)
  sync images(1,    stat=sync_status                      )
  sync images(1,                      errmsg=error_message)
  sync images(1                                           )

  sync images([1],  stat=sync_status, errmsg=error_message)
  sync images([1],  stat=sync_status                      )
  sync images([1],                    errmsg=error_message)
  sync images([1]                                         )

  !___ non-standard-conforming statement ___

  !______ invalid image sets ______

  ! Image set shall not depend on the value of stat-variable
  !ERROR: TBD
  sync images(sync_status, stat=sync_status)

  ! Image set shall not depend on the value of errmsg-variable
  !ERROR: TBD
  sync images(len(error_message), errmsg=error_message)

  ! Image set shall be a scalar or rank-1 array
  !ERROR: TBD
  sync images(invalid_rank)
 
  !______ invalid sync-stat-lists: invalid stat= ____________

  ! Invalid sync-stat-list keyword
  !ERROR: expected ')'
  sync images(1, status=sync_status)

  !ERROR: TBD
  sync images([1], stat=invalid_type)

  ! Stat-variable must an integer scalar
  !ERROR: TBD
  sync images(*, stat=non_scalar)
 
  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected ')'
  sync images(1, stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected ')'
  sync images([1], sync_status)

  !______ invalid sync-stat-lists: invalid errmsg= ____________

  ! Invalid errmsg-variable keyword
  !ERROR: expected ')'
  sync images(*, errormsg=error_message)

  !ERROR: TBD
  sync images(1, errmsg=invalid_type)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected ')'
  sync images([1], error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected ')'
  sync images(*, errmsg)

  !______ invalid sync-stat-lists: redundant sync-stat-list ____________

  ! No specifier shall appear more than once in a given sync-stat-list
  !ERROR: to be determined
  sync images(1, stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given sync-stat-list
  !ERROR: to be determined
  sync images([1], errmsg=error_message, errmsg=superfluous_errmsg)
 
  !______ invalid sync-stat-lists: coindexed stat-variable ____________

  ! Check constraint C1173 from the Fortran 2018 standard
  !ERROR: to be determined
  sync images(*, stat=coindexed_integer[1])
 
  ! Check constraint C1173 from the Fortran 2018 standard
  !ERROR: to be determined
  sync images(1, errmsg=coindexed_character[1])

end program test_sync_images
