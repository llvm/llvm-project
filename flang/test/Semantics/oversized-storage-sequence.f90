! RUN: %python %S/test_errors.py %s %flang_fc1

! The storage sequences laid out for COMMON blocks, EQUIVALENCE sets, and
! derived types must fit in a signed 64-bit byte size.
! Oversized standalone objects are left to the assembler and linker.

subroutine biggest_object
  ! huge(0_8) bytes exactly, the largest object that can be laid out
  character(len=9223372036854775807_8) :: c
end subroutine

subroutine biggest_array
  ! 2305843009213693951 * 4 == huge(0_8) - 3 bytes
  real :: a(2305843009213693951_8)
end subroutine

subroutine zero_sized_array
  ! a zero-sized array is not a wrapped size
  real :: z(2305843009213693952_8, 0)
end subroutine

subroutine explicit_bounds_that_fit
  ! extent = huge(0_8) in both cases, one byte per element
  integer(1) :: a(0_8:9223372036854775806_8)
  integer(1) :: b(-9223372036854775807_8:-1_8)
  common /fits1/ a
  common /fits2/ b
end subroutine

subroutine empty_explicit_bounds
  ! an upper bound below the lower bound makes the whole array empty
  integer(8) :: a(1_8:0_8)
  integer(8) :: b(-5_8:-10_8)
  integer(8) :: c(2305843009213693952_8, 1_8:0_8)
  common /empty/ a, b, c
end subroutine

subroutine unassociated_objects
  ! A procedure scope is not laid out as one storage sequence.
  integer(8) :: a(576460752303423488_8), b(576460752303423488_8), &
      c(576460752303423488_8)
  ! Standalone object sizes are also accepted here.
  integer(8) :: d(1152921504606846976_8)
  integer(8) :: e(2305843009213693952_8)
  character(kind=4, len=2305843009213693952_8) :: f
end subroutine

subroutine object_size_in_common
  ! 2305843009213693952 * 8 bytes is a multiple of 2**64, so the size folds to
  ! exactly zero; without a diagnostic 'b' would silently overlap 'a'
  integer(8) :: a(2305843009213693952_8)
  integer(8) :: b(4)
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a, b
end subroutine

subroutine element_size_in_common
  ! the size of a single element does not fit: 4 * 2305843009213693952 bytes
  character(kind=4, len=2305843009213693952_8) :: c
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ c
end subroutine

subroutine element_size_wraps_positive
  ! 4 * (2**62 + 1) wraps around to 4 when folded as a signed 64-bit
  ! integer, but the true element size does not fit
  character(kind=4, len=4611686018427387905_8) :: c
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ c
end subroutine

subroutine zero_lower_bound_in_common
  ! extent = huge(0_8) + 1 wraps around to a negative value, which must not be
  ! taken for an empty dimension; without a diagnostic 'b' would overlap 'a'
  integer(1) :: a(0_8:9223372036854775807_8)
  integer(1) :: b(4)
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a, b
end subroutine

subroutine negative_bounds_in_common
  ! extent = huge(0_8) fits, but there are two bytes per element
  integer(2) :: a(-9223372036854775807_8:-1_8)
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a
end subroutine

subroutine negative_to_positive_bounds_in_common
  ! extent = 2**63 wraps around to a negative value
  integer(1) :: a(-4611686018427387904_8:4611686018427387903_8)
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a
end subroutine

subroutine multidimensional_in_common
  ! 2**32 * 2**32 == 2**64 elements, so the size folds to exactly zero
  integer(1) :: a(4294967296_8, 4294967296_8)
  integer(1) :: b(4)
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a, b
end subroutine

subroutine multidimensional_bounds_in_common
  ! same element count, spelled with explicit zero lower bounds
  integer(1) :: a(0_8:4294967295_8, 0_8:4294967295_8)
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a
end subroutine

module derived_type_size
  !ERROR: The size of derived type 't' exceeds the maximum supported size of 9223372036854775807 bytes
  type t
    sequence
    integer(8) :: a(576460752303423488_8), b(576460752303423488_8), &
        c(576460752303423488_8)
  end type
  !ERROR: The size of derived type 'padded' exceeds the maximum supported size of 9223372036854775807 bytes
  type padded
    ! the components occupy exactly huge(0_8) bytes, but rounding that up to a
    ! multiple of the alignment of the type no longer fits
    sequence
    integer(8) :: n
    character(len=9223372036854775799_8) :: c
  end type
end module

subroutine equivalence_block_size
  ! the EQUIVALENCE chain extends the storage sequence to 3 * 2**62 bytes; the
  ! accumulated offsets are what can wrap around into a plausible small value
  integer(8) :: a(576460752303423488_8), b(576460752303423488_8)
  !ERROR: The size of the storage sequence created by EQUIVALENCE with 'c' exceeds the maximum supported size of 9223372036854775807 bytes
  integer(8) :: c(576460752303423488_8)
  equivalence (a(576460752303423488_8), b(1))
  equivalence (b(576460752303423488_8), c(1))
end subroutine

subroutine equivalence_oversized_member
  ! The base of a storage sequence is picked for layout reasons and is often
  ! its smallest member, so name the member that does not fit instead: here
  ! 'marker' is the base, but 'oversized' is what the user has to shrink.
  !ERROR: The size of the storage sequence created by EQUIVALENCE with 'oversized' exceeds the maximum supported size of 9223372036854775807 bytes
  integer(1) :: oversized(0_8:9223372036854775807_8)
  integer(8) :: marker
  equivalence (oversized, marker)
end subroutine

subroutine equivalence_base_size
  ! List the small object first so that the oversized array is selected as the
  ! base of the EQUIVALENCE storage sequence.
  !ERROR: The size of the storage sequence created by EQUIVALENCE with 'a' exceeds the maximum supported size of 9223372036854775807 bytes
  integer(8) :: a(2305843009213693952_8)
  integer(8) :: b
  equivalence (b, a(1))
end subroutine

subroutine equivalence_block_placement
  ! EQUIVALENCE block extents must fit independently of their scope offsets.
  character(len=9223372036854775800_8) :: c1
  character(len=8) :: c2
  equivalence (c1(9223372036854775793_8:), c2)
  integer :: a(2), b(2)
  equivalence (a(1), b(1))
end subroutine

subroutine equivalence_block_size_in_common
  integer(8) :: a(576460752303423488_8), b(576460752303423488_8), &
      c(576460752303423488_8)
  equivalence (a(576460752303423488_8), b(1))
  equivalence (b(576460752303423488_8), c(1))
  !ERROR: The size of COMMON block /blk/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /blk/ a
end subroutine
