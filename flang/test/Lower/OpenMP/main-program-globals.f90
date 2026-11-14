! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %if flangrt-modules %{ %flang_fc1 -fopenmp -emit-hlfir %t/intrinsics.f90 -o - | FileCheck %s --check-prefix=INTRINSICS-ON %}
! RUN: %if flangrt-modules %{ %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/intrinsics.f90 -o - | FileCheck %s --check-prefix=INTRINSICS-OFF --implicit-check-not='fir.global internal @_QFE' %}
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/miniweather.f90 -o %t.on
! RUN: FileCheck %s --check-prefixes=ON,ZERO < %t.on
! RUN: %flang_fc1 -fopenmp -fno-init-global-zero -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefixes=ON,UNDEF
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.alloca tuple' < %t.on
! RUN: bbc -fopenmp -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=ON
! RUN: bbc -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp-simd -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -fopenacc -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -x cuda -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: bbc -fopenmp -fopenacc -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: bbc -fopenmp -fcuda -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/objects.f90 -o %t.objects
! RUN: FileCheck %s --check-prefix=OBJECTS < %t.objects
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(array|matrix|storage|i|j|ap|ptr|record|private_value|listed|eq1|eq2|x|y|serial_only|x_private_i32|y_firstprivate_i32|x_linear_i32|allocated|address|pointee|element|extent|field|lastprivate_value|local_value|init_value)[[:space:]]}}' < %t.objects
! RUN: %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/objects.f90 -o %t.off
! RUN: FileCheck %s --check-prefix=BOUNDARY < %t.off
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(text|if_condition|if_action|where_value|forall_value)[[:space:]]}}' < %t.off
! RUN: FileCheck %s --check-prefix=EXPRESSIONS < %t.objects
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(element|extent|field|clause_only|header_only|statement_only)[[:space:]]}}' < %t.objects
! RUN: %flang_fc1 -emit-hlfir %t/objects.f90 -o %t.serial
! RUN: FileCheck %s --check-prefix=BOUNDARY < %t.serial
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(integer_value|real_value|complex_value|logical_value|text|if_condition|if_action|where_value|forall_value|direct)[[:space:]]}}' < %t.serial
! RUN: %flang_fc1 -fopenmp -mmlir -strict-fir-volatile-verifier -emit-hlfir %t/mixed.f90 -o %t.mixed
! RUN: FileCheck %s --check-prefix=MIXED < %t.mixed
! RUN: FileCheck %s --check-prefix=PROPERTIES < %t.mixed
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/callbacks.f90 -o %t.callbacks
! RUN: FileCheck %s --check-prefix=CALLBACK < %t.callbacks
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.alloca tuple' --implicit-check-not='fir.host_assoc' < %t.callbacks
! RUN: %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/callbacks.f90 -o - | FileCheck %s --check-prefix=BOX_OFF --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fcoarray -emit-hlfir %t/coarray.f90 -o %t.on
! RUN: FileCheck %s --check-prefix=COARRAY < %t.on
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFEa_coarray_handle{{[[:space:]]}}' < %t.on
! RUN: %flang_fc1 -fopenmp -fcoarray -emit-llvm %t/coarray.f90 -o - | FileCheck %s --check-prefix=LLVM
! MODULE: module

! MAIN scalars used in worker's PARALLEL region use globals and need no host tuple.
! ON-LABEL: func.func @_QQmain()
! ON-DAG: %[[MAIN_ADDR:[^ ]+]] = fir.address_of(@_QFEdx)
! ON-DAG: %[[MAIN_X:[^: ]+]]:2 = hlfir.declare %[[MAIN_ADDR]] {{.*}}uniq_name("_QFEdx")
! ON: fir.call @_FortranAioInputReal64(%{{[^,]+}}, %[[MAIN_X]]#0)
! ON: fir.call @_QFPworker()
! ON-LABEL: func.func private @_QFPworker()
! ON-DAG: %[[X_ADDR:[^ ]+]] = fir.address_of(@_QFEdx)
! ON-DAG: %[[X:[^: ]+]]:2 = hlfir.declare %[[X_ADDR]] {{.*}}uniq_name("_QFEdx")
! ON-DAG: fir.address_of(@_QFEdz)
! ON: omp.parallel
! ON: fir.load %[[X]]#0 : !fir.ref<f64>
! ON-DAG: fir.global internal @_QFEdx : f64
! ZERO: fir.zero_bits f64
! UNDEF: fir.undefined f64
! ON-DAG: fir.global internal @_QFEdz : f64
! OFF-LABEL: func.func @_QQmain()
! OFF-DAG: fir.alloca f64 {{.*}}bindc_name = "dx"
! OFF-DAG: fir.alloca f64 {{.*}}bindc_name = "dz"
! OFF: fir.alloca tuple
! OFF-LABEL: func.func private @_QFPworker(
! OFF-SAME: fir.host_assoc
!--- miniweather.f90
program p
integer :: nx,nz,i_beg,k_beg
real(8) :: dx,dz
nx=4
nz=3
i_beg=1
k_beg=1
read *,dx
read *,dz
call worker()
contains
subroutine worker()
integer :: i,k
real(8) :: a(4,3)
!$omp parallel do simd collapse(2) private(i,k)
do k=1,nz
 do i=1,nx
  a(i,k)=(i+i_beg)/dx+(k+k_beg)/dz
 end do
end do
!$omp end parallel do simd
print *,a
end subroutine
end program

! Independent call operands select globals; metadata and callee bodies do not.
! OBJECTS-LABEL: func.func @_QQmain()
! OBJECTS-DAG: fir.alloca i32 {{.*}}bindc_name = "serial_only"
! OBJECTS: omp.parallel
! OBJECTS-DAG: fir.global internal @_QFEif_condition : i32
! OBJECTS-DAG: fir.global internal @_QFEif_action : i32
! OBJECTS-DAG: fir.global internal @_QFEwhere_value : i32
! OBJECTS-DAG: fir.global internal @_QFEforall_value : i32
! OBJECTS-DAG: fir.global internal @_QFEinteger_value : i64
! OBJECTS-DAG: fir.global internal @_QFEreal_value : f64
! OBJECTS-DAG: fir.global internal @_QFEcomplex_value : complex<f64>
! OBJECTS-DAG: fir.global internal @_QFElogical_value : !fir.logical<4>
! OBJECTS-DAG: fir.global internal @_QFEtext : !fir.char<1,32>
! OBJECTS-DAG: fir.global internal @_QFElarge : !fir.char<1,33>
! OBJECTS-DAG: fir.global internal @_QFEsaved : i32
! BOUNDARY-DAG: fir.alloca i32 {{.*}}bindc_name = "direct"
! BOUNDARY-DAG: fir.alloca !fir.char<1,32> {{.*}}bindc_name = "text"
! BOUNDARY-DAG: fir.global internal @_QFElarge : !fir.char<1,33>
! EXPRESSIONS-LABEL: func.func @_QQmain()
! EXPRESSIONS: omp.parallel
! EXPRESSIONS: fir.call @_QFPsink(
! EXPRESSIONS: fir.call @_QFPassign(
! EXPRESSIONS-COUNT-17: fir.global internal @_QFE{{(direct|actual|rhs|lhs|call_arg|max_a|max_b|max_c|jn_lo|jn_hi|jn_arg|yn_lo|yn_hi|yn_arg|empty_index|argument_index|then_condition)}} :
! BOUNDARY-DAG: fir.global internal @_QFEsaved : i32
!--- objects.f90
program p
  integer(8) :: integer_value
  real(8) :: real_value
  complex(8) :: complex_value
  logical :: logical_value
  integer, save :: saved
  character(32) :: text
  character(33) :: large
  type t
    integer :: n
  end type
  type(t) :: record,value
  type handler; procedure(), pointer, nopass :: proc=>null(); end type
  type(handler) :: handlers(2)
  interface assignment(=)
    procedure assign,extract
  end interface
  integer :: direct,actual,element,extent,rhs,field,lhs,call_arg
  integer :: max_a,max_b,max_c,jn_lo,jn_hi,yn_lo,yn_hi,empty_index,argument_index
  integer :: then_condition,clause_only,header_only,statement_only,sf,arg,lastprivate_value,local_value,init_value
  real :: jn_arg,yn_arg
  integer :: array(2),matrix(2,2),private_value,listed,eq1,eq2,x,y
  integer :: if_condition,if_action,serial_only,where_value,forall_value,i,j
  integer :: x_private_i32,y_firstprivate_i32,x_linear_i32
  integer :: allocated,pointee,storage(1)
  integer(8) :: address
  pointer(address,pointee)
!$omp allocate(allocated)
  integer, allocatable :: ap
  integer, pointer :: ptr
  equivalence(eq1,eq2)
  namelist /group/ listed
  sf(arg)=statement_only+arg
  handlers(1)%proc=>no_arguments; handlers(2)%proc=>sink
  empty_index=1; argument_index=2; header_only=1; init_value=0
  direct=1; actual=2; element=3
  extent=4; rhs=5; field=6
  saved=1; storage=1; address=loc(storage); if_condition=1
  array=1; matrix=0; where_value=2; forall_value=3
!$omp parallel shared(clause_only) num_threads(header_only)
  print *,array,ap,ptr,record,private_value,listed,eq1,eq2
  print *,x_private_i32,y_firstprivate_i32,x_linear_i32
  print *,allocated,address,pointee,saved
  print *,direct,abs(actual),max(max_a,max_b,max_c),[element],[character(extent)::]
  value=t(field); print *,sf(1),lastprivate_value,local_value,init_value
  print *,bessel_jn(jn_lo,jn_hi,jn_arg),bessel_yn(yn_lo,yn_hi,yn_arg)
  call sink(actual)
  if (then_condition>0) then
    call handlers(argument_index)%proc(call_arg)
  end if
  value=rhs; lhs=value
  where(array>0)
    where(array<3) array=where_value
  end where
  forall(i=1:2)
    forall(j=1:2) matrix(i,j)=forall_value
  end forall
!$omp single
!$omp parallel if(.false.)
  print *,integer_value,real_value,complex_value,logical_value,text,large
!$omp end parallel
!$omp end single
  if (if_condition>0) if_action=1
!$omp end parallel
!$omp parallel private(private_value,x) firstprivate(y)
  private_value=x+y
!$omp end parallel
  x=0
!$omp simd linear(x) lastprivate(lastprivate_value)
  do i=1,4
    x=x+1; lastprivate_value=x
  end do
!$omp end simd
!$omp teams distribute parallel do
  do i=1,2
    call handlers(empty_index)%proc()
  end do
  do concurrent(i=1:1) local(local_value) local_init(init_value) reduce(+:direct); local_value=0; direct=direct+i; end do
  serial_only=1
contains
  subroutine no_arguments(); end subroutine
  subroutine extract(to,from); integer :: to; type(t) :: from; to=from%n; end subroutine
  subroutine sink(n)
    integer :: n
    print *,n
  end subroutine
  subroutine assign(to,from)
    type(t),intent(out) :: to
    integer,intent(in) :: from
    to%n=from
  end subroutine
end program

! Mixed captures use globals for x and a tuple for the array and VOLATILE v.
! MIXED-LABEL: func.func @_QQmain()
! MIXED: %[[A:[^: ]+]]:2 = hlfir.declare {{.*}}uniq_name("_QFEa")
! MIXED: %[[T:[^ ]+]] = fir.alloca tuple<{{.*}}!fir.box<!fir.array<2xi32>>{{.*}}>
! MIXED: %[[S:[^ ]+]] = fir.coordinate_of %[[T]], {{.*}} -> !fir.ref<!fir.box<!fir.array<2xi32>>>
! MIXED: %[[B:[^ ]+]] = fir.embox %[[A]]#0
! MIXED: fir.store %[[B]] to %[[S]]
! MIXED: fir.call @_QFPworker(%[[T]])
! MIXED-LABEL: func.func private @_QFPworker(
! MIXED-SAME: %[[H:[^: ]+]]:
! MIXED: %[[G:[^ ]+]] = fir.address_of(@_QFEalias)
! MIXED: %[[BYTES:[^ ]+]] = fir.convert %[[G]] : (!fir.ref<!fir.array<2xi32>>) -> !fir.ref<!fir.array<8xi8>>
! MIXED: %[[FOUR:[^ ]+]] = arith.constant 4 : index
! MIXED: %[[OFFSET:[^ ]+]] = fir.coordinate_of %[[BYTES]], %[[FOUR]]
! MIXED: %[[PTR:[^ ]+]] = fir.convert %[[OFFSET]]
! MIXED: %[[ALIAS:[^: ]+]]:2 = hlfir.declare %[[PTR]] storage(%[[BYTES]][4]) uniq_name("_QFEalias")
! MIXED: %[[TLS:[^: ]+]]:2 = hlfir.declare {{.*}}uniq_name("_QFEtls")
! MIXED: %[[ADDR:[^ ]+]] = omp.threadprivate %[[TLS]]#0
! MIXED: %[[LOCAL:[^: ]+]]:2 = hlfir.declare %[[ADDR]] uniq_name("_QFEtls")
! MIXED: %[[C:[^ ]+]] = fir.coordinate_of %[[H]], {{.*}} -> !fir.ref<!fir.box<!fir.array<2xi32>>>
! MIXED: %[[BOX:[^ ]+]] = fir.load %[[C]]
! MIXED: %[[ARRAY:[^ ]+]] = fir.box_addr %[[BOX]]
! MIXED: hlfir.declare %[[ARRAY]]({{.*}}) uniq_name("_QFEa") fortran_attrs<host_assoc>
! MIXED: fir.load %[[LOCAL]]#0
! MIXED: hlfir.assign {{.*}} to %[[LOCAL]]#0
! MIXED: omp.parallel
! MIXED: %[[THREAD:[^ ]+]] = omp.threadprivate %[[TLS]]#0
! MIXED: %[[PRIVATE:[^: ]+]]:2 = hlfir.declare %[[THREAD]] uniq_name("_QFEtls")
! MIXED: fir.load %[[ALIAS]]#0
! MIXED: fir.load %[[PRIVATE]]#0
! MIXED: fir.global internal @_QFEalias
! MIXED: %[[INIT:[^ ]+]] = fir.undefined !fir.array<2xi32>
! MIXED: %[[ELEVEN:[^ ]+]] = arith.constant 11 : i32
! MIXED: %[[FIRST:[^ ]+]] = fir.insert_value %[[INIT]], %[[ELEVEN]], [0 : index]
! MIXED: %[[TWENTYTWO:[^ ]+]] = arith.constant 22 : i32
! MIXED: %[[SECOND:[^ ]+]] = fir.insert_value %[[FIRST]], %[[TWENTYTWO]], [1 : index]
! MIXED: fir.has_value %[[SECOND]]
! PROPERTIES-LABEL: func.func private @_QFPworker(
! PROPERTIES-DAG: hlfir.declare {{.*}}uniq_name("_QFEav") fortran_attrs<asynchronous, internal_assoc>
! PROPERTIES-DAG: hlfir.declare {{.*}}uniq_name("_QFEv") fortran_attrs<volatile, host_assoc>
!--- mixed.f90
program p
  integer :: x,a(2),locality_only
  integer,volatile :: v
  integer,asynchronous :: av
  integer, save :: backing(2),alias,tls
  equivalence(backing(2),alias)
  data backing /11,22/
!$omp threadprivate(tls)
  x=7; v=1; av=2
  a=1; alias=2; tls=3
  call worker()
contains
  subroutine worker()
    integer :: i
    do concurrent(i=1:1) shared(locality_only); continue; end do
    tls=tls+1
!$omp parallel
    print *,x,a,alias,tls,v,av
!$omp end parallel
  end subroutine
end program

! Global captures supply direct calls, procedure boxes and CHARACTER arguments.
! CALLBACK-LABEL: func.func @_QQmain()
! CALLBACK: %[[P:[^: ]+]]:2 = hlfir.declare {{.*}}uniq_name("_QFEcallback")
! CALLBACK: %[[WORKER:[^ ]+]] = fir.address_of(@_QFPworker) : () -> ()
! CALLBACK: %[[BOX:[^ ]+]] = fir.emboxproc %[[WORKER]] :
! CALLBACK: fir.store %[[BOX]] to %[[P]]#0
! CALLBACK: %[[ACTUAL:[^ ]+]] = fir.address_of(@_QFPworker)
! CALLBACK: %[[ARG:[^ ]+]] = fir.emboxproc %[[ACTUAL]] :
! CALLBACK: fir.call @_QPinvoke(%[[ARG]])
! CALLBACK: omp.parallel
! CALLBACK: fir.call @_QFPworker()
! CALLBACK: %[[LOADED:[^ ]+]] = fir.load %[[P]]#0
! CALLBACK: %[[CODE:[^ ]+]] = fir.box_addr %[[LOADED]]
! CALLBACK: fir.call %[[CODE]]()
! CALLBACK-LABEL: func.func private @_QFPworker()
! CALLBACK: %[[TEXT_ADDR:[^ ]+]] = fir.address_of(@_QFEtext)
! CALLBACK: %[[LEN:[^ ]+]] = arith.constant 4 : index
! CALLBACK: %[[TEXT:[^: ]+]]:2 = hlfir.declare %[[TEXT_ADDR]] typeparams %[[LEN]]
! CALLBACK: %[[BYTES:[^ ]+]] = fir.convert %[[TEXT]]#0
! CALLBACK: %[[LENGTH:[^ ]+]] = fir.convert %[[LEN]] : (index) -> i64
! CALLBACK: fir.call @_FortranAioOutputAscii(%{{[^,]+}}, %[[BYTES]], %[[LENGTH]])
! CALLBACK-COUNT-2: fir.global internal @_QFE{{(text|result_len)}} :
! BOX_OFF: %[[HOST:[^ ]+]] = fir.alloca tuple<{{.*}}>
! BOX_OFF: %[[PTR:[^ ]+]] = fir.address_of(@_QFPworker)
! BOX_OFF: %[[BOX:[^ ]+]] = fir.emboxproc %[[PTR]], %[[HOST]]
! BOX_OFF: fir.store %[[BOX]]
! BOX_OFF: %[[ACTUAL:[^ ]+]] = fir.address_of(@_QFPworker)
! BOX_OFF: %[[ARG:[^ ]+]] = fir.emboxproc %[[ACTUAL]], %[[HOST]]
! BOX_OFF: fir.call @_QPinvoke(%[[ARG]])
! BOX_OFF: func.func private @_QFPworker({{.*}}fir.host_assoc
!--- callbacks.f90
program p
  interface
    subroutine invoke(proc)
      external :: proc
    end subroutine
  end interface
  procedure(), pointer :: callback
  integer :: x,result_len
  character(4) :: text
  x=7; result_len=4; text="abcd"
  callback=>worker
  call invoke(worker)
!$omp parallel num_threads(1)
  print *,x,text,result_len
  call worker()
  call callback()
!$omp end parallel
contains
  subroutine worker()
    print *,x,text,make()
  end subroutine
  function make() result(s); character(result_len) :: s; s=text; end function
end program
subroutine invoke(proc)
  external :: proc
  call proc()
end subroutine

! A scalar matching a coarray handle name needs separate storage.
! COARRAY-DAG: fir.alloca i32 {{.*}}bindc_name = "a_coarray_handle"
! COARRAY-DAG: fir.global internal @_QFEa_coarray_handle_extra : i32
! COARRAY-DAG: fir.global internal @_QFEordinary : i32
! LLVM: @_QFEa_coarray_handle = linkonce global %{{[^ ]*prif_coarray_handle_type}}
!--- coarray.f90
program p
  integer,allocatable :: a[:]
  integer :: a_coarray_handle,a_coarray_handle_extra,ordinary
  a_coarray_handle=17
  a_coarray_handle_extra=19
  ordinary=23
  allocate(a[*])
!$omp parallel num_threads(1)
  print *,a_coarray_handle,a_coarray_handle_extra,ordinary
!$omp end parallel
  deallocate(a)
end program

! Independent arguments distinguish user calls, inquiry metadata and folded IZEXT/JZEXT.
! INTRINSICS-ON-COUNT-5: fir.alloca {{.*}}bindc_name = "{{(meta|round_meta|keyword_meta|second_result|cpu_result)}}"
! INTRINSICS-ON-COUNT-8: fir.global internal @_QFE{{(actual|intrinsic_result|user_result|choice|optional_choice|izext_arg|jzext_arg|zext_sum)}} :
! INTRINSICS-OFF-COUNT-13: fir.alloca {{.*}}bindc_name = "{{(actual|intrinsic_result|user_result|choice|optional_choice|izext_arg|jzext_arg|zext_sum|meta|round_meta|keyword_meta|second_result|cpu_result)}}"
!--- intrinsics.f90
program p
  use, intrinsic :: ieee_arithmetic, only: true_value=>ieee_value, &
    ieee_positive_zero, rounding_support=>ieee_support_rounding, ieee_round_type, ieee_nearest, ieee_up
  real :: meta,actual,intrinsic_result,user_result,round_meta,keyword_meta,second_result,cpu_result
  integer(1) :: izext_arg,jzext_arg
  integer :: zext_sum,choice,optional_choice
  type(ieee_round_type) :: rounding,modes(2)
  rounding=ieee_nearest; modes=[ieee_nearest,ieee_up]
  meta=13; actual=6; round_meta=1; keyword_meta=2
  izext_arg=-1; jzext_arg=-1; choice=1; optional_choice=2
!$omp parallel num_threads(1)
  intrinsic_result=true_value(meta,ieee_positive_zero); user_result=ieee_value(actual)
  zext_sum=izext(izext_arg)+jzext(jzext_arg)
  print *,rounding_support(modes(choice),round_meta), &
    rounding_support(x=keyword_meta,round_value=rounding),rounding_support(modes(optional_choice))
  call second(second_result); call cpu_time(cpu_result)
!$omp end parallel
contains
  real function ieee_value(x); real :: x; ieee_value=x+1; end function
end program
