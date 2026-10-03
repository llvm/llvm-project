! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

! Same variant (p3) used with different 'construct' selector sets: not conforming.
subroutine r1_diff_sets
  !$omp declare variant (p1:p3) match (construct={parallel})
  !ERROR: Variant procedure 'p3' must have the same 'construct' selector set in all DECLARE VARIANT directives
  !$omp declare variant (p2:p3) match (construct={do})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! Same variant (p3) used with identical 'construct' selector sets: allowed.
subroutine r1_same_sets
  !$omp declare variant (p1:p3) match (construct={parallel})
  !$omp declare variant (p2:p3) match (construct={parallel})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! 'simd' (represented specially in the parse tree) appears in only one of the
! construct selector sets, so the sets differ: not conforming.
subroutine r1_simd_diff
  !$omp declare variant (p1:p3) match (construct={parallel})
  !ERROR: Variant procedure 'p3' must have the same 'construct' selector set in all DECLARE VARIANT directives
  !$omp declare variant (p2:p3) match (construct={parallel, simd})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! 'simd' appears in both construct selector sets: the sets are the same.
subroutine r1_simd_same
  !$omp declare variant (p1:p3) match (construct={parallel, simd})
  !$omp declare variant (p2:p3) match (construct={parallel, simd})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! 'simd' appears in both sets but with different simd properties (simdlen), so
! the construct selector sets differ: not conforming.
subroutine r1_simd_simdlen_diff
  !$omp declare variant (p1:p3) match (construct={simd(simdlen(4))})
  !ERROR: Variant procedure 'p3' must have the same 'construct' selector set in all DECLARE VARIANT directives
  !$omp declare variant (p2:p3) match (construct={simd(simdlen(8))})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! 'simd' with identical simd properties in both sets: allowed.
subroutine r1_simd_simdlen_same
  !$omp declare variant (p1:p3) match (construct={simd(simdlen(4))})
  !$omp declare variant (p2:p3) match (construct={simd(simdlen(4))})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! Bare 'simd' versus 'simd' with a property: the sets differ.
subroutine r1_simd_bare_vs_prop
  !$omp declare variant (p1:p3) match (construct={simd})
  !ERROR: Variant procedure 'p3' must have the same 'construct' selector set in all DECLARE VARIANT directives
  !$omp declare variant (p2:p3) match (construct={simd(simdlen(4))})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! 'simd' with inbranch versus notinbranch: the sets differ.
subroutine r1_simd_inbranch_diff
  !$omp declare variant (p1:p3) match (construct={simd(inbranch)})
  !ERROR: Variant procedure 'p3' must have the same 'construct' selector set in all DECLARE VARIANT directives
  !$omp declare variant (p2:p3) match (construct={simd(notinbranch)})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! The same simd properties written in a different order describe the same set:
! allowed (property order is not significant).
subroutine r1_simd_prop_reorder
  !$omp declare variant (p1:p3) match (construct={simd(simdlen(4), inbranch)})
  !$omp declare variant (p2:p3) match (construct={simd(inbranch, simdlen(4))})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! A procedure (p2) that is a variant may not later be used as a base function.
subroutine r3_variant_then_base
  !$omp declare variant (p1:p2) match (construct={parallel})
  !ERROR: The base procedure 'p2' is also specified as a variant procedure in another DECLARE VARIANT directive
  !$omp declare variant (p2:p3) match (construct={parallel})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! The same conflict is detected when the base use appears first: p1 is a base,
! then used as a variant.
subroutine r3_base_then_variant
  !$omp declare variant (p1:p2) match (construct={parallel})
  !ERROR: The variant procedure 'p1' is also specified as a base procedure in another DECLARE VARIANT directive
  !$omp declare variant (p3:p1) match (construct={parallel})
contains
  subroutine p1
  end subroutine
  subroutine p2
  end subroutine
  subroutine p3
  end subroutine
end subroutine

! Cross-function usage
module m_cross_function
contains
  subroutine p1
  end subroutine
  subroutine p2
    !$omp declare variant (p1) match (user={condition(.false.)},construct={do})
  end subroutine
  subroutine p3
    !ERROR: Variant procedure 'p1' must have the same 'construct' selector set in all DECLARE VARIANT directives
    !$omp declare variant (p1) match (user={condition(.true.)})
  end subroutine
end module

! Two directives for the same variant (p1) that both omit the 'construct'
! selector have the same (empty) construct selector set: allowed. Non-construct
! selectors (here 'user') do not affect the comparison.
module m_no_construct_ok
contains
  subroutine p1
  end subroutine
  subroutine p2
    !$omp declare variant (p1) match (user={condition(.true.)})
  end subroutine
  subroutine p3
    !$omp declare variant (p1) match (user={condition(.false.)})
  end subroutine
end module
