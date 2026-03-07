! This test checks lowering of OpenMP declare reduction Directive, with initialization
! via a subroutine.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
module maxtype_mod
  implicit none

  type maxtype
     integer::sumval
     integer::maxval
  end type maxtype

contains

  subroutine initme(x,n)
    type(maxtype) :: x,n
    x%sumval=0
    x%maxval=0
  end subroutine initme

  function mycombine(lhs, rhs)
    type(maxtype) :: lhs, rhs
    type(maxtype) :: mycombine
    mycombine%sumval = lhs%sumval + rhs%sumval
    mycombine%maxval = max(lhs%maxval, rhs%maxval)
  end function mycombine

  function func(x, n, init)
    type(maxtype) :: func
    integer :: n, i
    type(maxtype) :: x(n)
    type(maxtype) :: init
    type(maxtype) :: res
!$omp declare reduction(red_add_max:maxtype:omp_out=mycombine(omp_out,omp_in)) initializer(initme(omp_priv,omp_orig))
    res=init
!$omp simd reduction(red_add_max:res)
    do i=1,n
       res=mycombine(res,x(i))
    enddo
    func=res
  end function func

end module maxtype_mod
!CHECK:  omp.declare_reduction @red_add_max : !fir.ref<{{.*}}> attributes {byref_element_type = {{.*}}} alloc {
!CHECK:  %[[ALLOCA:.*]] = fir.alloca [[MAXTYPE:.*]]
!CHECK:  omp.yield(%[[ALLOCA]] : !fir.ref<[[MAXTYPE]]>)
!CHECK:  } init {
!CHECK:  ^bb0(%[[INIT_ARG0:.*]]: !fir.ref<[[MAXTYPE]]>, %[[INIT_ARG1:.*]]: !fir.ref<[[MAXTYPE]]>):
!CHECK:    %{{.*}} = fir.embox %[[INIT_ARG1]]
!CHECK:    %{{.*}} = fir.embox %[[INIT_ARG0]]
!CHECK:    %[[OMP_ORIG:.*]]:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_orig"} : (!fir.ref<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[OMP_PRIV:.*]]:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_priv"} : (!fir.ref<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    fir.call @_QMmaxtype_modPinitme(%[[OMP_PRIV]]#0, %[[OMP_ORIG]]#0) fastmath<contract> : (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>) -> ()
!CHECK:    %{{.*}} = fir.load %[[OMP_PRIV]]#0 : !fir.ref<[[MAXTYPE]]>
!CHECK:    omp.yield(%[[INIT_ARG1]] : !fir.ref<[[MAXTYPE]]>)
!CHECK:  } combiner {
!CHECK:  ^bb0(%[[LHS_ARG:.*]]: !fir.ref<[[MAXTYPE]]>, %[[RHS_ARG:.*]]: !fir.ref<[[MAXTYPE]]>):
!CHECK:    %[[RESULT:.*]] = fir.alloca [[MAXTYPE]] {bindc_name = ".result"}
!CHECK:    %[[OMP_IN:.*]]:2 = hlfir.declare %[[RHS_ARG]] {uniq_name = "omp_in"} : (!fir.ref<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[OMP_OUT:.*]]:2 = hlfir.declare %[[LHS_ARG]] {uniq_name = "omp_out"} : (!fir.ref<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[TMPRESULT:.*]]:2 = hlfir.declare %[[RESULT]] {uniq_name = ".tmp.func_result"} : (!fir.ref<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[COMBINE_RESULT:.*]] = fir.call @_QMmaxtype_modPmycombine(%[[OMP_OUT]]#0, %[[OMP_IN]]#0) fastmath<contract> : (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>) -> [[MAXTYPE]]
!CHECK:    fir.save_result %[[COMBINE_RESULT]] to %[[TMPRESULT]]#0 : [[MAXTYPE]], !fir.ref<[[MAXTYPE]]>
!CHECK:    %false = arith.constant false
!CHECK:    %[[EXPRRESULT:.*]] = hlfir.as_expr %[[TMPRESULT]]#0 move %false : (!fir.ref<[[MAXTYPE]]>, i1) -> !hlfir.expr<[[MAXTYPE]]>
!CHECK:    %[[ASSOCIATE:.*]]:3 = hlfir.associate %[[EXPRRESULT]] {adapt.valuebyref} : (!hlfir.expr<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>, i1)
!CHECK:    %[[RESULT_VAL:.*]] = fir.load %[[ASSOCIATE]]#0 : !fir.ref<[[MAXTYPE]]>
!CHECK:    hlfir.end_associate %[[ASSOCIATE]]#1, %[[ASSOCIATE]]#2 : !fir.ref<[[MAXTYPE]]>, i1
!CHECK:    hlfir.destroy %[[EXPRRESULT]] : !hlfir.expr<[[MAXTYPE]]>
!CHECK:    fir.store %[[RESULT_VAL]] to %[[LHS_ARG]] : !fir.ref<[[MAXTYPE]]>
!CHECK:    omp.yield(%[[LHS_ARG]] : !fir.ref<[[MAXTYPE]]>)
!CHECK:  }

!CHECK:  func.func @_QMmaxtype_modPinitme(%[[X_ARG:.*]]: !fir.ref<[[MAXTYPE]]> {fir.bindc_name = "x"}, %[[N_ARG:.*]]: !fir.ref<[[MAXTYPE]]> {fir.bindc_name = "n"}) {
!CHECK:    %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:    %[[N_DECL:.*]]:2 = hlfir.declare %[[N_ARG]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QMmaxtype_modFinitmeEn"} : (!fir.ref<[[MAXTYPE]]>, !fir.dscope) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ARG]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QMmaxtype_modFinitmeEx"} : (!fir.ref<[[MAXTYPE]]>, !fir.dscope) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[ZERO_0:.*]] = arith.constant 0 : i32
!CHECK:    %[[X_DESIGNATE_SUMVAL:.*]] = hlfir.designate %[[X_DECL]]#0{"sumval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    hlfir.assign %[[ZERO_0]] to %[[X_DESIGNATE_SUMVAL]] : i32, !fir.ref<i32>
!CHECK:    %[[ZERO_1:.*]] = arith.constant 0 : i32
!CHECK:    %[[X_DESIGNATE_MAXVAL:.*]] = hlfir.designate %[[X_DECL]]#0{"maxval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    hlfir.assign %[[ZERO_1]] to %[[X_DESIGNATE_MAXVAL]] : i32, !fir.ref<i32>
!CHECK:    return
!CHECK:  }


!CHECK:  func.func @_QMmaxtype_modPmycombine(%[[LHS:.*]]: !fir.ref<[[MAXTYPE]]> {fir.bindc_name = "lhs"}, %[[RHS:.*]]: !fir.ref<[[MAXTYPE]]> {fir.bindc_name = "rhs"}) -> [[MAXTYPE]] {
!CHECK:    %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:    %[[LHS_DECL:.*]]:2 = hlfir.declare %[[LHS]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QMmaxtype_modFmycombineElhs"} : (!fir.ref<[[MAXTYPE]]>, !fir.dscope) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[RESULT_ALLOC:.*]] = fir.alloca [[MAXTYPE]] {bindc_name = "mycombine", uniq_name = "_QMmaxtype_modFmycombineEmycombine"}
!CHECK:    %[[RESULT_DECL:.*]]:2 = hlfir.declare %[[RESULT_ALLOC]] {uniq_name = "_QMmaxtype_modFmycombineEmycombine"} : (!fir.ref<[[MAXTYPE]]>) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[RHS_DECL:.*]]:2 = hlfir.declare %[[RHS]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QMmaxtype_modFmycombineErhs"} : (!fir.ref<[[MAXTYPE]]>, !fir.dscope) -> (!fir.ref<[[MAXTYPE]]>, !fir.ref<[[MAXTYPE]]>)
!CHECK:    %[[LHS_DESIGNATE_SUMVAL:.*]] = hlfir.designate %[[LHS_DECL]]#0{"sumval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    %[[LHS_SUMVAL:.*]] = fir.load %[[LHS_DESIGNATE_SUMVAL]] : !fir.ref<i32>
!CHECK:    %[[RHS_DESIGNATE_SUMVAL:.*]] = hlfir.designate %[[RHS_DECL]]#0{"sumval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    %[[RHS_SUMVAL:.*]] = fir.load %[[RHS_DESIGNATE_SUMVAL]] : !fir.ref<i32>
!CHECK:    %[[SUM:.*]] = arith.addi %[[LHS_SUMVAL]], %[[RHS_SUMVAL]] : i32
!CHECK:    %[[RESULT_DESIGNATE_SUMVAL:.*]] = hlfir.designate %[[RESULT_DECL]]#0{"sumval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    hlfir.assign %[[SUM]] to %[[RESULT_DESIGNATE_SUMVAL]] : i32, !fir.ref<i32>
!CHECK:    %[[LHS_DESIGNATE_MAXVAL:.*]] = hlfir.designate %[[LHS_DECL]]#0{"maxval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    %[[LHS_MAXVAL:.*]] = fir.load %[[LHS_DESIGNATE_MAXVAL]] : !fir.ref<i32>
!CHECK:    %[[RHS_DESIGNATE_MAXVAL:.*]] = hlfir.designate %[[RHS_DECL]]#0{"maxval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    %[[RHS_MAXVAL:.*]] = fir.load %[[RHS_DESIGNATE_MAXVAL]] : !fir.ref<i32>
!CHECK:    %[[CMP:.*]] = arith.cmpi sgt, %[[LHS_MAXVAL]], %[[RHS_MAXVAL]] : i32
!CHECK:    %[[MAX_VAL:.*]] = arith.select %[[CMP]], %[[LHS_MAXVAL]], %[[RHS_MAXVAL]] : i32
!CHECK:    %[[RESULT_DESIGNAGE_MAXVAL:.*]] = hlfir.designate %[[RESULT_DECL]]#0{"maxval"}   : (!fir.ref<[[MAXTYPE]]>) -> !fir.ref<i32>
!CHECK:    hlfir.assign %[[MAX_VAL]] to %[[RESULT_DESIGNAGE_MAXVAL]] : i32, !fir.ref<i32>
!CHECK:    %[[RESULT:.*]] = fir.load %[[RESULT_DECL]]#0 : !fir.ref<[[MAXTYPE]]>
!CHECK:    return %[[RESULT]] : [[MAXTYPE]]
!CHECK:  }
