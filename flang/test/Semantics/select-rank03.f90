! RUN: %python %S/test_errors.py %s %flang_fc1
program test
  real, allocatable :: a0, a1(:)
  real, pointer :: p0, p1(:)
  real, target :: t0, t1(1)
 contains
  subroutine allocatables(a)
    real, allocatable :: a(..)
    !ERROR: RANK (*) cannot be used when selector is POINTER or ALLOCATABLE
    select rank(a)
    rank (0)
      allocate(a) ! ok
      deallocate(a) ! ok
      allocate(a, source=a0) ! ok
      allocate(a, mold=p0) ! ok
      a = 1. ! ok
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
      a = [1.]
      !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
      allocate(a, source=a1)
      allocate(a, mold=p1) ! ok, mold= ignored
    rank (1)
      allocate(a(1)) ! ok
      deallocate(a) ! ok
      a = 1. ! ok
      a = [1.] ! ok
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a, source=a0)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a, mold=p0)
      allocate(a, source=a1) ! ok
      allocate(a, mold=p1) ! ok
    rank (2)
      allocate(a(1,1)) ! ok
      deallocate(a) ! ok
      a = 1. ! ok
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 2 array of REAL(4) and rank 1 array of REAL(4)
      a = [1.]
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a, source=a0)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a, mold=p0)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a, source=a1)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a, mold=p1)
    rank (*)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a)
      deallocate(a)
      a = 1.
    rank default
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(a)
      deallocate(a)
      a = 1.
    end select
  end
  subroutine pointers(p)
    real, pointer :: p(..)
    !ERROR: RANK (*) cannot be used when selector is POINTER or ALLOCATABLE
    select rank(p)
    rank (0)
      allocate(p) ! ok
      deallocate(p) ! ok
      allocate(p, source=a0) ! ok
      allocate(p, mold=p0) ! ok
      !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
      allocate(p, source=a1)
      allocate(p, mold=p1) ! ok, mold ignored
      p => t0 ! ok
      !ERROR: Pointer has rank 0 but target has rank 1
      p => t1
    rank (1)
      allocate(p(1)) ! ok
      deallocate(p) ! ok
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p, source=a0)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p, mold=p0)
      allocate(p, source=a1) ! ok
      allocate(p, mold=p1) ! ok
      !ERROR: Pointer has rank 1 but target has rank 0
      p => t0
      p => t1 ! ok
    rank (2)
      allocate(p(1,1)) ! ok
      deallocate(p) ! ok
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p, source=a0)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p, mold=p0)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p, source=a1)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p, mold=p1)
      !ERROR: Pointer has rank 2 but target has rank 0
      p => t0
      !ERROR: Pointer has rank 2 but target has rank 1
      p => t1
    rank (*)
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p)
      deallocate(p)
    rank default
      !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
      allocate(p)
      deallocate(p)
      !ERROR: pointer 'p' associated with object 't0' with incompatible type or shape
      p => t0
      !ERROR: pointer 'p' associated with object 't1' with incompatible type or shape
      p => t1
    end select
  end
end
