! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_sel
!----------------------

! CHECK-LABEL: vec_sel_testi1
subroutine vec_sel_testi1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[arg1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[arg2]], %[[arg3]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK:  %[[comp:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK:  %[[and1:.*]] = and <16 x i8> %[[arg1]], %[[comp]]
! CHECK:  %[[and2:.*]] = and <16 x i8> %[[arg2]], %[[arg3]]
! CHECK:  %{{[0-9]+}} = or <16 x i8> %[[and1]], %[[and2]]
end subroutine vec_sel_testi1

! CHECK-LABEL: vec_sel_testi2
subroutine vec_sel_testi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1, arg2, r
  vector(unsigned(2)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <8 x i16> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <8 x i16> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <8 x i16> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <8 x i16>
end subroutine vec_sel_testi2

! CHECK-LABEL: vec_sel_testi4
subroutine vec_sel_testi4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1, arg2, r
  vector(unsigned(4)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <4 x i32> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <4 x i32> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <4 x i32> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <4 x i32>
end subroutine vec_sel_testi4

! CHECK-LABEL: vec_sel_testi8
subroutine vec_sel_testi8(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  vector(unsigned(8)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <2 x i64> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <2 x i64> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <2 x i64> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <2 x i64>
end subroutine vec_sel_testi8

! CHECK-LABEL: vec_sel_testu1
subroutine vec_sel_testu1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[arg1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[arg2]], %[[arg3]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.or %[[and1:.*]], %[[and2]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK:  %[[comp:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK:  %[[and1:.*]] = and <16 x i8> %[[arg1]], %[[comp]]
! CHECK:  %[[and2:.*]] = and <16 x i8> %[[arg2]], %[[arg3]]
! CHECK:  %{{[0-9]+}} = or <16 x i8> %[[and1]], %[[and2]]
end subroutine vec_sel_testu1

! CHECK-LABEL: vec_sel_testu2
subroutine vec_sel_testu2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1, arg2, r
  vector(unsigned(2)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1:.*]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <8 x i16> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <8 x i16> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <8 x i16> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <8 x i16>
end subroutine vec_sel_testu2

! CHECK-LABEL: vec_sel_testu4
subroutine vec_sel_testu4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1, arg2, r
  vector(unsigned(4)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <4 x i32> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <4 x i32> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <4 x i32> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <4 x i32>
end subroutine vec_sel_testu4

! CHECK-LABEL: vec_sel_testu8
subroutine vec_sel_testu8(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  vector(unsigned(8)) :: arg3
  r = vec_sel(arg1, arg2, arg3)
  
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <2 x i64> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <2 x i64> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <2 x i64> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <2 x i64>
end subroutine vec_sel_testu8

! CHECK-LABEL: vec_sel_testr4
subroutine vec_sel_testr4(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  vector(unsigned(4)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <4 x float> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <4 x float> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <4 x i32> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <4 x float>
end subroutine vec_sel_testr4

! CHECK-LABEL: vec_sel_testr8
subroutine vec_sel_testr8(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  vector(unsigned(8)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[varg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[bcv1:.*]] = vector.bitcast %[[varg1]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[bcv2:.*]] = vector.bitcast %[[varg2]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[bcv3:.*]] = vector.bitcast %[[varg3]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[bcv3]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %[[and1:.*]] = arith.andi %[[bcv1]], %[[xor]] : vector<16xi8>
! CHECK-FIR: %[[and2:.*]] = arith.andi %[[bcv2]], %[[bcv3]] : vector<16xi8>
! CHECK-FIR: %[[or:.*]] = arith.ori %[[and1]], %[[and2]] : vector<16xi8>
! CHECK-FIR: %[[bcor:.*]] = vector.bitcast %[[or]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcor]] : (vector<2xf64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[bc3:.*]] = llvm.bitcast %[[arg3]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[bc3:.*]], %[[c]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and1:.*]] = llvm.and %[[bc1]], %[[xor]]  : vector<16xi8>
! CHECK-LLVMIR: %[[and2:.*]] = llvm.and %[[bc2]], %[[bc3]]  : vector<16xi8>
! CHECK-LLVMIR: %[[or:.*]] = llvm.or %[[and1]], %[[and2]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[or]] : vector<16xi8> to vector<2xf64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <2 x double> %5 to <16 x i8>
! CHECK: %[[bc2:.*]] = bitcast <2 x double> %6 to <16 x i8>
! CHECK: %[[bc3:.*]] = bitcast <2 x i64> %7 to <16 x i8>
! CHECK: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! CHECK: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! CHECK: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! CHECK: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <2 x double>
end subroutine vec_sel_testr8
