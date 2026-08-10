! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for errors in sync images statements

program test_sync_images
  implicit none

  integer sync_status, me
  character(len=128) error_message

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

  !ERROR: expected '('
  sync images

  !______ invalid sync-stat-lists: invalid stat= ____________

  ! Invalid sync-stat-list keyword
  !ERROR: expected ')'
  sync images(1, status=sync_status)

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

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected ')'
  sync images([1], error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected ')'
  sync images(*, errmsg)

end program test_sync_images
