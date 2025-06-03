!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! The DEVICE and TARGET_DEVICE trait sets

subroutine f00
  !$omp metadirective &
!ERROR: DEVICE_NUM is not a valid trait for DEVICE trait set
  !$omp & when(device={device_num(10)}: nothing)
end

subroutine f01
  !$omp metadirective &
!This is ok: all traits are valid
  !$omp & when(device={arch("some-arch"), isa("some-isa"), kind("some-kind")}:&
  !$omp & nothing)
end

subroutine f02
  !$omp metadirective &
!This is ok: all traits are valid
  !$omp & when(target_device={arch("some-arch"), device_num(10), &
  !$omp & isa("some-isa"), kind("some-kind"), uid("some-uid")}: nothing)
end

subroutine f03
  !$omp metadirective &
!This is ok: extension traits are allowed
  !$omp & when(device={some_new_trait}: nothing)
end

subroutine f04
  !$omp metadirective &
!This is ok: extension traits are allowed
  !$omp & when(target_device={another_new_trait(12, 21)}: nothing)
end

