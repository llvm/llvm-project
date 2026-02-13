
// RUN: fir-opt --split-input-file --cuf-device-global %s | FileCheck %s


module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>

  gpu.module @cuda_device_mod {
  }
}

// CHECK: gpu.module @cuda_device_mo
// CHECK-NEXT: fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>

// -----

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMm1ECb(dense<[90, 100, 110]> : tensor<3xi32>) constant : !fir.array<3xi32>
  fir.global @_QMm2ECc(dense<[100, 200, 300]> : tensor<3xi32>) constant : !fir.array<3xi32>
}

// CHECK: fir.global @_QMm1ECb
// CHECK: fir.global @_QMm2ECc
// CHECK: gpu.module @cuda_device_mod
// CHECK-DAG: fir.global @_QMm2ECc
// CHECK-DAG: fir.global @_QMm1ECb

// -----

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMmEddarrays {data_attr = #cuf.cuda<managed>} : !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTdevicearrays{phi_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_i:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMmTdevicearrays{phi_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_i:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) {allocator_idx = 3 : i32} : (!fir.heap<!fir.array<?x!fir.type<_QMmTdevicearrays{phi_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_i:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTdevicearrays{phi_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_i:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmTdevicearrays{phi_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,phi0_i:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_r:!fir.box<!fir.heap<!fir.array<?xf64>>>,buf_i:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>>>
  }
  fir.global linkonce_odr @_QMmE.dt.devicearrays constant target : !fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}> {
    %0 = fir.undefined !fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>
    fir.has_value %0 : !fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>
  }
}


// CHECK-NAG: fir.global @_QMmEddarrays
// CHECK-NAG: fir.global linkonce_odr @_QMmE.dt.devicearrays
// CHECK: gpu.module @cuda_device_mod
// CHECK-NAG: fir.global @_QMmEddarrays
// CHECK-NAG: fir.global linkonce_odr @_QMmE.dt.devicearrays
