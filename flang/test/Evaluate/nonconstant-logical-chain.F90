! RUN: %flang_fc1 -fsyntax-only %s
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
