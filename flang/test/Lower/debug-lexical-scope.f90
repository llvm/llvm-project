! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo -I %t -module-dir %t %t/test.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=line-tables-only -mmlir --mlir-print-debuginfo -I %t -module-dir %t %t/test.f90 -o - | FileCheck %s --check-prefix=NODEBUG
! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo -I %t -module-dir %t %t/test.f90 -o - | FileCheck %s --check-prefix=NODEBUG

! Test the lexical scope fused into the location of the code of each branch of
! a construct that associates names, when variables are described. The
! positions are those in the files that split-file extracts below.

! CHECK-LABEL: func.func @_QMmPtypes(
! CHECK: hlfir.declare %arg0 {{.*}}uniq_name("_QMmFtypesEp"){{.*}} loc(#[[DUMMY_P:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFtypesEp"){{.*}} loc(#[[TYPE_IS:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFtypesEq"){{.*}} loc(#[[CLASS_IS:loc[0-9]+]])
! CHECK: %[[LINE:[a-z0-9_]+]] = arith.constant 15 : i32
! CHECK-NEXT: fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %[[LINE]]) {{.*}} loc(#[[CLASS_IS_PRINT:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFtypesEp"){{.*}} loc(#[[CLASS_DEFAULT:loc[0-9]+]])
! CHECK: %[[AFTER_LINE:[a-z0-9_]+]] = arith.constant 20 : i32
! CHECK-NEXT: fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %[[AFTER_LINE]]) {{.*}} loc(#[[AFTER:loc[0-9]+]])

! CHECK-LABEL: func.func @_QMmPranks(
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFranksEx"){{.*}} loc(#[[RANK0:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFranksEx"){{.*}} loc(#[[RANK1:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFranksEx"){{.*}} loc(#[[RANK_DEFAULT:loc[0-9]+]])

! CHECK-LABEL: func.func @_QMmPassoc(
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFassocEa"){{.*}} loc(#[[ASSOC_A:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFassocEb"){{.*}} loc(#[[ASSOC_B:loc[0-9]+]])

! CHECK-LABEL: func.func @_QMmPinc(
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFincEp"){{.*}} loc(#[[INC_TYPE_IS:loc[0-9]+]])
! CHECK: hlfir.declare %{{[0-9#]+}} uniq_name("_QMmFincEp"){{.*}} loc(#[[INC_CLASS_DEFAULT:loc[0-9]+]])

! The declaration of a dummy argument is in no branch.
! CHECK-DAG: #[[DUMMY_P]] = loc("{{.*}}test.f90":10:{{[0-9]+}})

! The scope of a branch starts at its statement and covers the declarations of
! the names that it associates. The scope of a branch of a nested construct has
! the scope of the branch that holds the construct as parent.
! CHECK-DAG: #[[TYPE_IS]] = loc(fused<#fir.lexical_scope<loc = #[[TYPE_IS_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[TYPE_IS_STMT]] = loc("{{.*}}test.f90":12:5)
! CHECK-DAG: #[[CLASS_IS]] = loc(fused<#fir.lexical_scope<loc = #[[CLASS_IS_STMT:loc[0-9]+]], parent = <loc = #[[TYPE_IS_STMT]]>>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[CLASS_IS_STMT]] = loc("{{.*}}test.f90":14:7)
! CHECK-DAG: #[[CLASS_DEFAULT]] = loc(fused<#fir.lexical_scope<loc = #[[CLASS_DEFAULT_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[CLASS_DEFAULT_STMT]] = loc("{{.*}}test.f90":17:5)

! The scope covers the other code of the branch, whose runtime calls still get
! the line of their statement. The code after the construct is in no branch.
! CHECK-DAG: #[[CLASS_IS_PRINT]] = loc(fused<#fir.lexical_scope<loc = #[[CLASS_IS_STMT]], parent = <loc = #[[TYPE_IS_STMT]]>>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[AFTER]] = loc("{{.*}}test.f90":20:5)

! CHECK-DAG: #[[RANK0]] = loc(fused<#fir.lexical_scope<loc = #[[RANK0_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[RANK0_STMT]] = loc("{{.*}}test.f90":26:5)
! CHECK-DAG: #[[RANK1]] = loc(fused<#fir.lexical_scope<loc = #[[RANK1_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[RANK1_STMT]] = loc("{{.*}}test.f90":28:5)
! CHECK-DAG: #[[RANK_DEFAULT]] = loc(fused<#fir.lexical_scope<loc = #[[RANK_DEFAULT_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[RANK_DEFAULT_STMT]] = loc("{{.*}}test.f90":30:5)

! The names that one statement associates share a scope.
! CHECK-DAG: #[[ASSOC_A]] = loc(fused<#fir.lexical_scope<loc = #[[ASSOC_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[ASSOC_B]] = loc(fused<#fir.lexical_scope<loc = #[[ASSOC_STMT]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[ASSOC_STMT]] = loc("{{.*}}test.f90":37:5)

! The scope of a branch in an included file starts in that file.
! CHECK-DAG: #[[INC_TYPE_IS]] = loc(fused<#fir.lexical_scope<loc = #[[INC_TYPE_IS_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[INC_TYPE_IS_STMT]] = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>[#[[INC_TYPE_IS_POS:loc[0-9]+]], #{{loc[0-9]+}}])
! CHECK-DAG: #[[INC_TYPE_IS_POS]] = loc("{{.*}}construct.inc":2:1)
! CHECK-DAG: #[[INC_CLASS_DEFAULT]] = loc(fused<#fir.lexical_scope<loc = #[[INC_CLASS_DEFAULT_STMT:loc[0-9]+]]>>[#{{loc[0-9]+}}])
! CHECK-DAG: #[[INC_CLASS_DEFAULT_STMT]] = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>[#[[INC_CLASS_DEFAULT_POS:loc[0-9]+]], #{{loc[0-9]+}}])
! CHECK-DAG: #[[INC_CLASS_DEFAULT_POS]] = loc("{{.*}}construct.inc":4:1)

! NODEBUG-NOT: lexical_scope

!--- test.f90
module m
  type :: base
    integer :: a = 1
  end type
  type, extends(base) :: derived
    integer :: b = 2
  end type
contains
  subroutine types(p, q)
    class(base), intent(in) :: p, q
    select type (p)
    type is (derived)
      select type (q)
      class is (derived)
        print *, p%b, q%b
      end select
    class default
      print *, p%a
    end select
    print *, p%a
  end subroutine types

  subroutine ranks(x)
    integer, intent(in) :: x(..)
    select rank (x)
    rank (0)
      print *, x
    rank (1)
      print *, x(1)
    rank default
      print *, rank(x)
    end select
  end subroutine ranks

  subroutine assoc(n)
    integer, intent(in) :: n
    associate (a => n * 2, b => n * 3)
      print *, a, b
    end associate
  end subroutine assoc

  subroutine inc(p)
    class(base), intent(in) :: p
    include "construct.inc"
  end subroutine inc
end module m

!--- construct.inc
select type (p)
type is (derived)
  print *, p%b
class default
  print *, p%a
end select
