! The "use_device_addr" was added to the "target data" directive in OpenMP 5.0.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! This tests primary goal is to check the promotion of non-CPTR arguments from
! use_device_ptr to use_device_addr works, without breaking any functionality.

!CHECK: func.func @{{.*}}only_use_device_ptr()
!CHECK: omp.target_data use_device_addr(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) use_device_ptr(%{{.*}} -> %{{.*}} : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
subroutine only_use_device_ptr
    use iso_c_binding
    integer, pointer, dimension(:) :: array
    real, pointer :: pa(:)
    type(c_ptr) :: cptr

       !$omp target data use_device_ptr(pa, cptr, array)
       !$omp end target data
     end subroutine

!CHECK: func.func @{{.*}}mix_use_device_ptr_and_addr()
!CHECK: omp.target_data use_device_addr(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) use_device_ptr({{.*}} : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
subroutine mix_use_device_ptr_and_addr
    use iso_c_binding
    integer, pointer, dimension(:) :: array
    real, pointer :: pa(:)
    type(c_ptr) :: cptr

       !$omp target data use_device_ptr(pa, cptr) use_device_addr(array)
       !$omp end target data
     end subroutine

     !CHECK: func.func @{{.*}}only_use_device_addr()
     !CHECK: omp.target_data use_device_addr(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
     subroutine only_use_device_addr
        use iso_c_binding
        integer, pointer, dimension(:) :: array
        real, pointer :: pa(:)
        type(c_ptr) :: cptr

       !$omp target data use_device_addr(pa, cptr, array)
       !$omp end target data
     end subroutine

     !CHECK: func.func @{{.*}}mix_use_device_ptr_and_addr_and_map()
     !CHECK: omp.target_data map_entries(%{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) use_device_addr(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) use_device_ptr(%{{.*}} : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
     subroutine mix_use_device_ptr_and_addr_and_map
        use iso_c_binding
        integer :: i, j
        integer, pointer, dimension(:) :: array
        real, pointer :: pa(:)
        type(c_ptr) :: cptr

       !$omp target data use_device_ptr(pa, cptr) use_device_addr(array) map(tofrom: i, j)
       !$omp end target data
     end subroutine

     !CHECK: func.func @{{.*}}only_use_map()
     !CHECK: omp.target_data map_entries(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
     subroutine only_use_map
        use iso_c_binding
        integer, pointer, dimension(:) :: array
        real, pointer :: pa(:)
        type(c_ptr) :: cptr

       !$omp target data map(pa, cptr, array)
       !$omp end target data
     end subroutine
