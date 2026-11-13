! RUN: %flang_fc1 -fsyntax-only -Wno-folding-exception %s
! Oversized runtime constructors can remain unfolded. In particular, wrapping
! the byte size or element count must not allow an enormous scalar expansion.

subroutine byte_size_overflow(x)
  type array
    complex values(2305843009213693953_8)
  end type
  type(array) x
  x = array((1.0, 2.0))
end

subroutine element_count_overflow(x)
  type array
    complex values(2147483648_8, 2147483648_8, 4)
  end type
  type(array) x
  x = array((1.0, 2.0))
end
