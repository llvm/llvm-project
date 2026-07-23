!RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
module m
  type impure_t
   contains
    final :: finalize
  end type
  type inner_t
    type(impure_t) :: impure
   contains
    procedure :: set => inner_set
  end type
  type outer_t
    type(inner_t) :: inner
  end type
  interface
    module subroutine finalize(this)
      type(impure_t), intent(inout) :: this
    end
    pure module subroutine inner_set(this)
      class(inner_t), intent(inout) :: this
    end
  end interface
 contains
  pure subroutine test(outer)
    type(outer_t), intent(inout) :: outer
    !WARNING: 'inner' has impure FINAL procedure 'finalize' and must be definable in this pure context
    call outer%inner%set()
  end
end
