! RUN: %flang_fc1 -fsyntax-only %s
! Compile-time smoke test only: no timing threshold is enforced, so this
! test also passes without the copy-reduction optimization.
! For a manual before/after benchmark, set FLANG_BEFORE and FLANG_AFTER to
! equivalently configured compiler builds and run from the same checkout:
!   chain_test=flang/test/Evaluate/nonconstant-logical-chain.F90
!   time "$FLANG_BEFORE" -fc1 -fsyntax-only "$chain_test"
!   time "$FLANG_AFTER" -fc1 -fsyntax-only "$chain_test"
! Repeat the commands to account for startup, filesystem, and host-load effects.
! Keep the operands nonconstant so folding cannot collapse the expression
! tree while semantics analyzes its intermediate expressions.
#define REPEAT16(x) x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x.or.x
#define REPEAT256(x) REPEAT16(REPEAT16(x))
subroutine character_chain(keyword, found)
  character(*), intent(in) :: keyword
  logical, intent(out) :: found
  found = REPEAT256((keyword == 'match'))
end subroutine

subroutine integer_chain(k, found)
  integer, intent(in) :: k
  logical, intent(out) :: found
  found = REPEAT256((k == 42))
end subroutine

subroutine logical_chain(b, found)
  logical, intent(in) :: b(:)
  logical, intent(out) :: found
  found = REPEAT256(b(1))
end subroutine
