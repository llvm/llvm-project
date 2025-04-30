! @@name:	standalone.1f
! @@type:	F-free
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
SUBROUTINE STANDALONE_WRONG()
  INTEGER  A
  A = 1
  ! the FLUSH directive must not be the action statement
  ! in an IF statement
  IF (A .NE. 0) !$OMP FLUSH(A) ! { error "PGF90-S-0034-Syntax error at or near end of line" }

  ! the BARRIER directive must not be the action statement
  ! in an IF statement
  IF (A .NE. 0) !$OMP BARRIER ! { error "PGF90-S-0034-Syntax error at or near end of line" }

  ! the TASKWAIT directive must not be the action statement
  ! in an IF statement
  IF (A .NE. 0) !$OMP TASKWAIT ! { error "PGF90-S-0034-Syntax error at or near end of line" }

  ! the TASKYIELD directive must not be the action statement
  ! in an IF statement
  IF (A .NE. 0) !$OMP TASKYIELD ! { error "PGF90-S-0034-Syntax error at or near end of line" }

  GOTO 100

  ! the FLUSH directive must not be a labeled branch target
  ! statement
  100 !$OMP FLUSH(A) ! { error "PGF90-S-0018-Illegal label 100 - must be followed by a keyword or an identifier" }
  GOTO 200

  ! the BARRIER directive must not be a labeled branch target
  ! statement
  200 !$OMP BARRIER ! { error "PGF90-S-0018-Illegal label 200 - must be followed by a keyword or an identifier" }
  GOTO 300

  ! the TASKWAIT directive must not be a labeled branch target
  ! statement
  300 !$OMP TASKWAIT ! { error "PGF90-S-0018-Illegal label 300 - must be followed by a keyword or an identifier" }
  GOTO 400

  ! the TASKYIELD directive must not be a labeled branch target
  ! statement
  400 !$OMP TASKYIELD ! { error "PGF90-S-0018-Illegal label 400 - must be followed by a keyword or an identifier" }

END SUBROUTINE
