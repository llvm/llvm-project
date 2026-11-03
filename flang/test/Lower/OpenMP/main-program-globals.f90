! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/miniweather.f90 -o %t.on
! RUN: FileCheck %s --check-prefix=ON < %t.on
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.alloca tuple' < %t.on
! RUN: bbc -fopenmp -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=ON
! RUN: bbc -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -fopenmp-is-target-device -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp-simd -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -fopenacc -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %flang_fc1 -fopenmp -x cuda -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: bbc -fopenmp -fopenacc -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: bbc -fopenmp -fcuda -emit-hlfir %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OFF --implicit-check-not='fir.global internal @_QFE'
! RUN: %if aarch64-registered-target %{ %flang_fc1 -triple aarch64-unknown-linux-gnu -O3 -ffast-math -fopenmp -emit-llvm %t/miniweather.f90 -o - | FileCheck %s --check-prefix=OPT %}
! RUN: %if asserts %{ %flang_fc1 -fopenmp -fdebug-dump-pft -mllvm -debug-only=flang-pft %t/miniweather.f90 2>&1 | FileCheck %s --check-prefix=PFT_ON %}
! RUN: %if asserts %{ %flang_fc1 -fopenmp -fno-openmp-main-program-globals -fdebug-dump-pft -mllvm -debug-only=flang-pft %t/miniweather.f90 2>&1 | FileCheck %s --check-prefix=PFT_OFF %}
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/objects.f90 -o %t.objects
! RUN: FileCheck %s --check-prefix=OBJECTS < %t.objects
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(array|matrix|storage|i|j|ap|ptr|record|private_value|listed|eq1|eq2|x|y|serial_only|x_private_i32|y_firstprivate_i32|x_linear_i32|allocated|address|pointee|actual|element|extent|rhs|field)[[:space:]]}}' < %t.objects
! RUN: %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/objects.f90 -o %t.off
! RUN: FileCheck %s --check-prefix=BOUNDARY < %t.off
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(text|if_condition|if_action|where_value|forall_value)[[:space:]]}}' < %t.off
! RUN: FileCheck %s --check-prefix=EXPRESSIONS < %t.objects
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(actual|element|extent|rhs|field)[[:space:]]}}' < %t.objects
! RUN: %flang_fc1 -emit-hlfir %t/objects.f90 -o %t.serial
! RUN: FileCheck %s --check-prefix=BOUNDARY < %t.serial
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFE{{(integer_value|real_value|complex_value|logical_value|text|if_condition|if_action|where_value|forall_value|direct)[[:space:]]}}' < %t.serial
! RUN: %flang_fc1 -fopenmp -mllvm -strict-fir-volatile-verifier -emit-hlfir %t/mixed.f90 -o %t.mixed
! RUN: FileCheck %s --check-prefix=MIXED < %t.mixed
! RUN: FileCheck %s --check-prefix=PROPERTIES < %t.mixed
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/callbacks.f90 -o %t.callbacks
! RUN: FileCheck %s --check-prefix=CALLBACK < %t.callbacks
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.alloca tuple' --implicit-check-not='fir.host_assoc' < %t.callbacks
! RUN: %flang_fc1 -fopenmp -fno-openmp-main-program-globals -emit-hlfir %t/callbacks.f90 -o - | FileCheck %s --check-prefix=BOX_OFF --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -emit-fir %t/callbacks.f90 -o - | fir-opt --flang-licm | FileCheck %s --check-prefix=LICM
! RUN: %flang_fc1 -fopenmp -fcoarray -emit-hlfir %t/coarray.f90 -o %t.on
! RUN: FileCheck %s --check-prefix=COARRAY < %t.on
! RUN: FileCheck %s --check-prefix=MODULE --implicit-check-not='fir.global internal @_QFEa_coarray_handle{{[[:space:]]}}' < %t.on
! RUN: %flang_fc1 -fopenmp -fcoarray -emit-llvm %t/coarray.f90 -o - | FileCheck %s --check-prefix=LLVM
! MODULE: module

! Check globals and removal of the host tuple for MAIN scalars referenced in an
! internal procedure's PARALLEL region. Disabled modes retain legacy storage;
! OPT checks reciprocal computation before the outlined loop.
! ON-LABEL: func.func @_QQmain()
! ON: fir.call @_QFPworker()
! ON-LABEL: func.func private @_QFPworker()
! ON-DAG: fir.address_of(@_QFEdx)
! ON-DAG: fir.address_of(@_QFEdz)
! ON: omp.parallel
! ON-DAG: fir.global internal @_QFEdx : f64
! ON-DAG: fir.global internal @_QFEdz : f64
! OFF-LABEL: func.func @_QQmain()
! OFF-DAG: fir.alloca f64 {{.*}}bindc_name = "dx"
! OFF-DAG: fir.alloca f64 {{.*}}bindc_name = "dz"
! OFF: fir.alloca tuple
! OFF-LABEL: func.func private @_QFPworker(
! OFF-SAME: fir.host_assoc
! OPT-LABEL: define internal void @_QFPworker..omp_par(
! OPT: omp_collapsed.body.lr.ph:
! OPT-DAG: %[[DX:[^ ]+]] = load double, ptr @_QFEdx
! OPT-DAG: %[[DZ:[^ ]+]] = load double, ptr @_QFEdz
! OPT-DAG: %[[RX:[^ ]+]] = fdiv fast double 1.000000e+00, %[[DX]]
! OPT-DAG: %[[RZ:[^ ]+]] = fdiv fast double 1.000000e+00, %[[DZ]]
! OPT: br label %[[BODY:omp_collapsed.body]]
! OPT: [[BODY]]:
! OPT-NOT: fdiv
! OPT: fmul fast double {{.*}}, %[[RX]]
! OPT-NOT: fdiv
! OPT: fmul fast double {{.*}}, %[[RZ]]
! OPT-NOT: fdiv
! OPT: br i1
! PFT_ON: {{^main-global symbol$}}
! PFT_OFF-NOT: {{^main-global }}
! PFT_OFF: Program
! PFT_OFF-NOT: {{^main-global }}
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

! Select intrinsic scalars in nested regions and statement expressions while
! preserving excluded objects and private copies. Generated privatizer names
! must not collide with excluded scalars. Calls and constructors do not
! seed selection. The 32/33-byte CHARACTER cases distinguish new selection from
! existing static storage. serial_only checks collection stops at END PARALLEL.
! OBJECTS-LABEL: func.func @_QQmain()
! OBJECTS: fir.alloca i32 {{.*}}bindc_name = "serial_only"
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
! EXPRESSIONS: fir.global internal @_QFEdirect : i32
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
  interface assignment(=)
    procedure assign
  end interface
  integer :: direct,actual,element,extent,rhs,field
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
  direct=1; actual=2; element=3
  extent=4; rhs=5; field=6
  saved=1; storage=1; address=loc(storage); if_condition=1
  array=1; matrix=0; where_value=2; forall_value=3
!$omp parallel
  print *,array,ap,ptr,record,private_value,listed,eq1,eq2
  print *,x_private_i32,y_firstprivate_i32,x_linear_i32
  print *,allocated,address,pointee,saved
  print *,direct,abs(actual),[element],[character(extent)::]
  print *,t(field)
  call sink(actual)
  value=rhs
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
!$omp simd linear(x)
  do i=1,4
    x=x+1
  end do
!$omp end simd
  serial_only=1
contains
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

! Keep the array in the host tuple while promoted scalars use globals. Preserve
! EQUIVALENCE offsets and DATA initialization, threadprivate bindings, and
! VOLATILE/ASYNCHRONOUS attributes.
! MIXED-LABEL: func.func @_QQmain()
! MIXED: %[[A:[^: ]+]]:2 = hlfir.declare {{.*}}uniq_name("_QFEa")
! MIXED: %[[T:[^ ]+]] = fir.alloca tuple<!fir.box<!fir.array<2xi32>>>
! MIXED: %[[S:[^ ]+]] = fir.coordinate_of %[[T]],
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
! MIXED: %[[C:[^ ]+]] = fir.coordinate_of %[[H]],
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
! PROPERTIES-DAG: hlfir.declare {{.*}}uniq_name("_QFEv") fortran_attrs<volatile, internal_assoc>
!--- mixed.f90
program p
  integer :: x,a(2)
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
    tls=tls+1
!$omp parallel
    print *,x,a,alias,tls,v,av
!$omp end parallel
  end subroutine
end program

! Removing the last host capture also updates procedure boxes and direct and
! indirect calls. Loads must stay in loops that call a modifying sibling.
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
! BOX_OFF: %[[HOST:[^ ]+]] = fir.alloca tuple<!fir.ref<i32>>
! BOX_OFF: %[[PTR:[^ ]+]] = fir.address_of(@_QFPworker)
! BOX_OFF: %[[BOX:[^ ]+]] = fir.emboxproc %[[PTR]], %[[HOST]]
! BOX_OFF: fir.store %[[BOX]]
! BOX_OFF: %[[ACTUAL:[^ ]+]] = fir.address_of(@_QFPworker)
! BOX_OFF: %[[ARG:[^ ]+]] = fir.emboxproc %[[ACTUAL]], %[[HOST]]
! BOX_OFF: fir.call @_QPinvoke(%[[ARG]])
! BOX_OFF: func.func private @_QFPworker({{.*}}fir.host_assoc
! LICM-LABEL: func.func @_QQmain()
! LICM: %[[MAIN_X:[^ ]+]] = fir.declare {{.*}}uniq_name("_QFEx") fortran_attrs<internal_assoc>
! LICM: fir.do_loop
! LICM-NOT: {{^ *[}]}}
! LICM: fir.call @_QFPbump(
! LICM-NOT: {{^ *[}]}}
! LICM: fir.load %[[MAIN_X]] :
! LICM: {{^ *[}]}}
! LICM-LABEL: func.func private @_QFPaccumulate(
! LICM: %[[X:[^ ]+]] = fir.declare {{.*}}uniq_name("_QFEx") fortran_attrs<internal_assoc>
! LICM: fir.do_loop
! LICM-NOT: {{^ *[}]}}
! LICM: fir.call @_QFPbump(
! LICM-NOT: {{^ *[}]}}
! LICM: fir.load %[[X]] :
! LICM: {{^ *[}]}}
!--- callbacks.f90
program p
  interface
    subroutine invoke(proc)
      external :: proc
    end subroutine
  end interface
  procedure(), pointer :: callback
  integer :: x,i,s
  x=7
  callback=>worker
  call invoke(worker)
!$omp parallel num_threads(1)
  print *,x
  call worker()
  call callback()
!$omp end parallel
  x=0
  call accumulate()
  x=0
  s=0
  do i=1,3
    call bump()
    s=s+x
  end do
  print *,s
contains
  subroutine worker()
    print *,x
  end subroutine
  subroutine accumulate()
    integer :: i,s
    s=0
    do i=1,3
      call bump()
      s=s+x
    end do
    print *,s
  end subroutine
  subroutine bump()
    x=x+1
  end subroutine
end program
subroutine invoke(proc)
  external :: proc
  call proc()
end subroutine

! Preserve the scalar whose name would collide with a coarray handle; a longer
! suffix and an ordinary scalar remain eligible for global storage.
! COARRAY: fir.alloca i32 {{.*}}bindc_name = "a_coarray_handle"
! COARRAY-DAG: fir.global internal @_QFEa_coarray_handle_extra : i32
! COARRAY-DAG: fir.global internal @_QFEordinary : i32
! LLVM: @_QFEa_coarray_handle = linkonce global %{{[^ ]*prif_coarray_handle_type}}
! LLVM: call void @llvm.memmove{{.*}}(ptr @_QFEa_coarray_handle, ptr {{.*}}, i64 8, i1 false)
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
