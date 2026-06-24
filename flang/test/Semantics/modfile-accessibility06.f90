! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests F2023 8.6.1 p2: module-name accessibility with USE renames.
! Exercises rename at the re-export layer and at the consuming USE site.

module origin
  implicit none
  integer :: velocity = 100
  integer :: altitude = 200
end module

! relay_alpha re-exports origin's symbols under their original names.
module relay_alpha
  use origin
end module

! relay_beta re-exports origin::velocity under the name 'speed'.
module relay_beta
  use origin, only: speed => velocity
end module

! relay_gamma re-exports origin's symbols under their original names
! (same as relay_alpha, for multi-source tests).
module relay_gamma
  use origin
end module

! Case 1: Same ultimate via rename -- both PRIVATE.
! relay_alpha exports 'velocity', relay_beta exports 'speed'.
! These are different local names, so no collision; each gets its own
! module's accessibility independently.
module rename_both_private
  use relay_alpha, only: velocity
  use relay_beta, only: speed
  implicit none
  private relay_alpha
  private relay_beta
end module

! Case 2: Same ultimate via rename -- mixed access on separate names.
! velocity comes from relay_alpha (PRIVATE), speed comes from relay_beta
! (PUBLIC).  No collision; each reflects its source module's access.
module rename_mixed_separate
  use relay_alpha, only: velocity
  use relay_beta, only: speed
  implicit none
  private relay_alpha
  public relay_beta
end module

! Case 3: Consumer-side rename merges two sources under one local name.
! Both relay_alpha::velocity and relay_beta::speed have the same ultimate
! (origin::velocity).  The consumer renames speed back to 'velocity',
! so both contribute to local name 'velocity'.
! relay_alpha is PRIVATE, relay_gamma is PUBLIC -> PUBLIC wins.
module rename_consumer_merge
  use relay_alpha, only: velocity
  use relay_gamma, only: velocity
  implicit none
  private relay_alpha
  public relay_gamma
end module

! Case 4: Consumer-side rename merges two sources, both PRIVATE.
module rename_consumer_both_private
  use relay_alpha, only: velocity
  use relay_gamma, only: velocity
  implicit none
  private relay_alpha
  private relay_gamma
end module

! Case 5: Consumer-side rename with partial access-stmt.
! relay_alpha is PRIVATE, relay_gamma has no access-stmt.
! Not all contributing modules are named -> default PUBLIC applies.
module rename_consumer_partial
  use relay_alpha, only: velocity
  use relay_gamma, only: velocity
  implicit none
  private relay_alpha
end module

! Case 6: Re-export rename brought into consumer under yet another name.
! relay_beta exports 'speed' (renamed from origin::velocity).
! Consumer renames 'speed' to 'rate'.  Both PRIVATE -> PRIVATE.
module rename_chain_private
  use relay_alpha, only: rate => velocity
  use relay_beta, only: rate => speed
  implicit none
  private relay_alpha
  private relay_beta
end module

! Case 7: Re-export rename chain, mixed access -> PUBLIC wins.
module rename_chain_mixed
  use relay_alpha, only: rate => velocity
  use relay_beta, only: rate => speed
  implicit none
  private relay_alpha
  public relay_beta
end module

program test_renames
  implicit none

  ! Case 1: separate names, both PRIVATE.
  block
    !ERROR: 'velocity' is PRIVATE in 'rename_both_private'
    use rename_both_private, only: velocity
  end block
  block
    !ERROR: 'speed' is PRIVATE in 'rename_both_private'
    use rename_both_private, only: speed
  end block

  ! Case 2: separate names, mixed access.
  ! velocity from relay_alpha (PRIVATE), speed from relay_beta (PUBLIC).
  block
    !ERROR: 'velocity' is PRIVATE in 'rename_mixed_separate'
    use rename_mixed_separate, only: velocity
  end block
  block
    use rename_mixed_separate, only: speed  ! PUBLIC, no error
  end block

  ! Case 3: consumer merge under one name, PUBLIC wins (no error).
  block
    use rename_consumer_merge, only: velocity
  end block

  ! Case 4: consumer merge under one name, both PRIVATE.
  block
    !ERROR: 'velocity' is PRIVATE in 'rename_consumer_both_private'
    use rename_consumer_both_private, only: velocity
  end block

  ! Case 5: consumer merge, partial access-stmt -> default PUBLIC (no error).
  block
    use rename_consumer_partial, only: velocity
  end block

  ! Case 6: rename chain, both PRIVATE.
  block
    !ERROR: 'rate' is PRIVATE in 'rename_chain_private'
    use rename_chain_private, only: rate
  end block

  ! Case 7: rename chain, mixed -> PUBLIC wins (no error).
  block
    use rename_chain_mixed, only: rate
  end block
end program
