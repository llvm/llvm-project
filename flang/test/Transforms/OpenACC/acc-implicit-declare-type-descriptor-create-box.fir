// RUN: fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-declare)" %s | FileCheck %s

// A fir.create_box that builds a descriptor for an array of a derived type
// references that type's runtime type descriptor (.dt.) global. The
// acc-implicit-declare pass must discover that global through the
// IndirectGlobalAccess interface and mark it so it is made available on the
// device, exactly as it does for fir.embox / fir.rebox.

module {
  // Stand-in for the type descriptor global emitted for the derived type. The
  // pass only looks it up by name; its contents are irrelevant here.
  fir.global linkonce_odr @_QMmmE.dt.struct constant target : i64 {
    %0 = arith.constant 0 : i64
    fir.has_value %0 : i64
  }

  func.func @test(%arg0: !fir.ref<!fir.array<?x!fir.type<_QMmmTstruct{member:f32}>>>, %lb: index, %ext: index, %sm: index) {
    acc.serial {
      %0 = fir.create_box %arg0 lbs(%lb) extents(%ext) strides(%sm) : (!fir.ref<!fir.array<?x!fir.type<_QMmmTstruct{member:f32}>>>, index, index, index) -> !fir.box<!fir.array<?x!fir.type<_QMmmTstruct{member:f32}>>>
      acc.yield
    }
    return
  }
}

// CHECK: fir.global {{.*}}@_QMmmE.dt.struct {acc.declare = #acc.declare<dataClause = acc_copyin>}{{.*}}constant target
