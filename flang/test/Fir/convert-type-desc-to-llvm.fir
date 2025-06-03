// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMDAT
// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMDAT
// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=i386-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMDAT
// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=powerpc64le-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMDAT
// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=x86_64-pc-win32 type-descriptors-renamed-for-assembly=true" %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMDAT
// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=aarch64-apple-darwin type-descriptors-renamed-for-assembly=true" %s | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMDAT
// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=amdgcn-amd-amdhsa type-descriptors-renamed-for-assembly=1 datalayout=e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-P0" %s | FileCheck -check-prefixes=CHECK %s

// Check descriptor for a derived type. Check that the f18Addendum flag is set
// to 1 meaning the addendum is present (true) and the addendum values are
// inserted.

fir.global linkonce @_QMtest_dinitE.dt.tseq constant : i8

func.func @embox1(%arg0: !fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) {
  %0 = fir.embox %arg0() : (!fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
  return
}

// CHECK-COMDAT: llvm.mlir.global linkonce constant @_QMtest_dinitEXdtXtseq() comdat(@__llvm_comdat::@_QMtest_dinitEXdtXtseq) {addr_space = 0 : i32} : i8
// CHECK-NO-COMDAT: llvm.mlir.global linkonce constant @_QMtest_dinitEXdtXtseq() {addr_space = 0 : i32} : i8
// CHECK-LABEL: llvm.func @embox1
// CHECK:         %[[TYPE_CODE:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK:         %[[VERSION:.*]] = llvm.mlir.constant(20240719 : i32) : i32
// CHECK:         %{{.*}} = llvm.insertvalue %[[VERSION]], %{{.*}}[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, ptr, array<1 x i64>)> 
// CHECK:         %[[TYPE_CODE_I8:.*]] = llvm.trunc %[[TYPE_CODE]] : i32 to i8
// CHECK:         %{{.*}} = llvm.insertvalue %[[TYPE_CODE_I8]], %{{.*}}[4] : !llvm.struct<(ptr, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr, array<1 x i{{.*}}>)>
// CHECK:         %[[TDESC:.*]] = llvm.mlir.addressof @_QMtest_dinitEXdtXtseq : !llvm.ptr
// CHECK:         %{{.*}} = llvm.insertvalue %[[TDESC]], %{{.*}}[7] : !llvm.struct<(ptr, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr, array<1 x i{{.*}}>)>
