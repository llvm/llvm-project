! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
! REQUIRES: target=powerpc{{.*}}

! C: MainProgram scope: ppc_vec_types
! CHECK-LABEL: MainProgram scope: ppc_vec_types size={{[0-9]*}} alignment={{[0-9]*}}
program ppc_vec_types
  implicit none
  vector(integer(4)) :: vi
  vector(real(8)) :: vr
  vector(unsigned(2)) :: vu
  __vector_pair :: vp
  __vector_quad :: vq
! CHECK-DAG: vi size=16 offset={{[0-9]*}}: ObjectEntity type: vector(integer(4))
! CHECK-DAG: vr size=16 offset={{[0-9]*}}: ObjectEntity type: vector(real(8))
! CHECK-DAG: vu size=16 offset={{[0-9]*}}: ObjectEntity type: vector(unsigned(2))
! CHECK-DAG: vp size=32 offset={{[0-9]*}}: ObjectEntity type: __vector_pair
! CHECK-DAG: vq size=64 offset={{[0-9]*}}: ObjectEntity type: __vector_quad

contains
! CHECK-LABEL: Subprogram scope: test_vec_integer_func size={{[0-9]*}} alignment={{[0-9]*}}
  function test_vec_integer_func(arg1)
    vector(integer(4)) :: arg1
    vector(integer(4)) :: test_vec_integer_func
! CHECK-DAG: arg1 size=16 offset={{[0-9]*}}: ObjectEntity dummy type: vector(integer(4))
! CHECK-DAG: test_vec_integer_func size=16 offset={{[0-9]*}}: ObjectEntity funcResult type: vector(integer(4))
  end function test_vec_integer_func

! CHECK-LABEL: Subprogram scope: test_vec_real_func size={{[0-9]*}} alignment={{[0-9]*}}
  function test_vec_real_func(arg1)
    vector(real(8)) :: arg1
    vector(real(8)) :: test_vec_real_func
! CHECK-DAG: arg1 size=16 offset={{[0-9]*}}: ObjectEntity dummy type: vector(real(8))
! CHECK-DAG: test_vec_real_func size=16 offset={{[0-9]*}}: ObjectEntity funcResult type: vector(real(8))
  end function test_vec_real_func

! CHECK-LABEL: Subprogram scope: test_vec_unsigned_func
  function test_vec_unsigned_func(arg1)
    vector(unsigned(2)) :: arg1
    vector(unsigned(2)) :: test_vec_unsigned_func
! CHECK-DAG: arg1 size=16 offset={{[0-9]*}}: ObjectEntity dummy type: vector(unsigned(2))
! CHECK-DAG: test_vec_unsigned_func size=16 offset={{[0-9]*}}: ObjectEntity funcResult type: vector(unsigned(2))
  end function test_vec_unsigned_func

! CHECK-LABEL: Subprogram scope: test_vec_pair_func
  function test_vec_pair_func(arg1)
    __vector_pair :: arg1
    __vector_pair :: test_vec_pair_func
! CHECK-DAG: arg1 size=32 offset={{[0-9]*}}: ObjectEntity dummy type: __vector_pair
! CHECK-DAG: test_vec_pair_func size=32 offset={{[0-9]*}}: ObjectEntity funcResult type: __vector_pair
  end function test_vec_pair_func

! CHECK-LABEL: Subprogram scope: test_vec_quad_func
  function test_vec_quad_func(arg1)
    __vector_quad :: arg1
    __vector_quad :: test_vec_quad_func
! CHECK-DAG: arg1 size=64 offset={{[0-9]*}}: ObjectEntity dummy type: __vector_quad
! CHECK-DAG: test_vec_quad_func size=64 offset={{[0-9]*}}: ObjectEntity funcResult type: __vector_quad
  end function test_vec_quad_func

end program ppc_vec_types
