// Test TBAA tags attachment for accesses of derived types
// that contain descriptors. They cannot be attached
// with any reasonable "data" tag by fir-add-alias-tags pass,
// because this will conflict with "any descriptor"
// tags attached to the descriptor accesses that may be part
// of these derived type values.
// Representative Fortran example:
// subroutine test
//   type t
//      real, allocatable :: f
//   end type t
//   associate(a => callee([1.0]))
//     call ext(a(1)%f)
//   end associate
// contains
//   elemental type(t) function callee(x)
//     real, intent(in) :: x
//     callee = t(x)
//   end function callee
// end subroutine test
// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @_QPtest() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.alloca !fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>> {bindc_name = ".tmp.array"}
  %1 = fir.convert %0 : (!fir.ref<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>) -> !fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>
  %2 = fir.alloca f32 {bindc_name = ".tmp"}
  %3 = fir.alloca !fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}> {bindc_name = ".result"}
  %4 = fir.dummy_scope : !fir.dscope
  %14 = fir.address_of(@_QQro.1xr4.0) : !fir.ref<!fir.array<1xf32>>
  %15 = fir.shape %c1 : (index) -> !fir.shape<1>
  %16 = fir.declare %14(%15) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1xr4.0"} : (!fir.ref<!fir.array<1xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<1xf32>>
  %17 = fir.declare %1(%15) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> !fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>
  fir.do_loop %arg0 = %c1 to %c1 step %c1 unordered {
    %27 = fir.array_coor %16(%15) %arg0 : (!fir.ref<!fir.array<1xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %28 = fir.declare %2 {uniq_name = ".tmp"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK: fir.load %{{.*}} {tbaa = {{.*}}} : !fir.ref<f32>
    %29 = fir.load %27 : !fir.ref<f32>
// CHECK: fir.store %{{.*}} {tbaa = {{.*}}} : !fir.ref<f32>
    fir.store %29 to %28 : !fir.ref<f32>
    %30 = fir.call @_QFtestPcallee(%28) proc_attrs<elemental, pure> fastmath<fast> : (!fir.ref<f32>) -> !fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>
    fir.save_result %30 to %3 : !fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>, !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
    %31 = fir.declare %3 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
    %32 = fir.array_coor %17(%15) %arg0 : (!fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
    %33 = fir.load %31 : !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
// CHECK: fir.store %{{.*}} to %{{[0-9]+}} : !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
    fir.store %33 to %32 : !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
  }
  %18 = fir.convert %17 : (!fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>) -> !fir.ref<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>
  %19 = fir.declare %18(%15) {uniq_name = "_QFtestEa"} : (!fir.ref<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>
  %20 = fir.array_coor %19(%15) %c1 : (!fir.ref<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>
  %21 = fir.coordinate_of %20, f : (!fir.ref<!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK: fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.heap<f32>>>
  %22 = fir.load %21 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %23 = fir.box_addr %22 : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %24 = fir.convert %23 : (!fir.heap<f32>) -> !fir.ref<f32>
  fir.call @_QPext(%24) fastmath<fast> : (!fir.ref<f32>) -> ()
  %25 = fir.embox %17(%15) : (!fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>>
  %26 = fir.convert %25 : (!fir.box<!fir.heap<!fir.array<1x!fir.type<_QFtestTt{f:!fir.box<!fir.heap<f32>>}>>>>) -> !fir.box<none>
  fir.call @_FortranADestroyWithoutFinalization(%26) : (!fir.box<none>) -> ()
  return
}
}
