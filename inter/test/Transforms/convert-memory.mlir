// RUN: inter-opt %s --inter-normalize-cf --inter-convert-calls --inter-convert-memory | FileCheck %s

module {
  // CHECK-LABEL: func.func @ptradd_aliases
  // CHECK: [[ENTRY:%.*]] = xw.token
  // CHECK: {{%.*}}, [[READ:%.*]] = xw.load {{%.*}} dep [[ENTRY]]
  // CHECK: xw.store {{.*}} dep [[ENTRY]]
  // CHECK: xw.store {{.*}} dep [[READ]]
  func.func @ptradd_aliases(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>,
                           %offset: i64, %value: i32) attributes {
      xemachine.kernel} {
    %pa = xw.ptradd %a, %offset : !llvm.ptr<1>, i64
    %pb = xw.ptradd %b, %offset : !llvm.ptr<1>, i64
    %loaded = llvm.load %pa : !llvm.ptr<1> -> i32
    llvm.store %value, %pb : i32, !llvm.ptr<1>
    llvm.store %value, %pa : i32, !llvm.ptr<1>
    return
  }

  // CHECK-LABEL: func.func @read_join
  // CHECK: [[ENTRY:%.*]] = xw.token
  // CHECK: {{%.*}}, [[READ0:%.*]] = xw.load {{%.*}} dep [[ENTRY]]
  // CHECK: {{%.*}}, [[READ1:%.*]] = xw.load {{%.*}} dep [[ENTRY]]
  // CHECK: [[READS:%.*]] = xw.token_join [[READ0]], [[READ1]]
  // CHECK: [[WRITE:%.*]] = xw.store {{.*}} dep [[READS]]
  // CHECK: xw.store {{.*}} dep [[ENTRY]]
  llvm.func spir_kernelcc @read_join(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>,
                                    %index: i64, %value: i32) {
    %pa = llvm.getelementptr %a[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %first = llvm.load %pa : !llvm.ptr<1> -> i32
    %second = llvm.load %pa : !llvm.ptr<1> -> i32
    llvm.store %value, %pa : i32, !llvm.ptr<1>
    %pb = llvm.getelementptr %b[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %value, %pb : i32, !llvm.ptr<1>
    llvm.return
  }

  // An unknown write aliases both kernel arguments and becomes the dependency
  // for later accesses to either one.
  // CHECK-LABEL: func.func @unknown_write
  // CHECK: [[ENTRY:%.*]] = xw.token
  // CHECK: {{%.*}}, [[READ0:%.*]] = xw.load {{%.*}} dep [[ENTRY]]
  // CHECK: {{%.*}}, [[READ1:%.*]] = xw.load {{%.*}} dep [[ENTRY]]
  // CHECK: [[READS:%.*]] = xw.token_join [[READ0]], [[READ1]]
  // CHECK: [[UNKNOWN:%.*]] = xw.store {{.*}} dep [[READS]]
  // CHECK: {{%.*}}, {{%.*}} = xw.load {{%.*}} dep [[UNKNOWN]]
  llvm.func spir_kernelcc @unknown_write(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>,
                                        %raw: i64, %index: i64, %value: i32) {
    %pa = llvm.getelementptr %a[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %first = llvm.load %pa : !llvm.ptr<1> -> i32
    %pb = llvm.getelementptr %b[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %second = llvm.load %pb : !llvm.ptr<1> -> i32
    %unknown = llvm.inttoptr %raw : i64 to !llvm.ptr<1>
    llvm.store %value, %unknown : i32, !llvm.ptr<1>
    %after = llvm.load %pa : !llvm.ptr<1> -> i32
    llvm.return
  }

  // CHECK-LABEL: func.func @atomic_and_barrier
  // CHECK: [[ENTRY:%.*]] = xw.token
  // CHECK: {{%.*}}, [[ATOMIC:%.*]] = xw.atomic_add {{.*}} dep [[ENTRY]]
  // CHECK: {{%.*}}, [[READ0:%.*]] = xw.load {{%.*}} dep [[ATOMIC]]
  // CHECK: {{%.*}}, [[READ1:%.*]] = xw.load {{%.*}} dep [[ENTRY]]
  // CHECK: [[READS:%.*]] = xw.token_join [[READ0]], [[READ1]]
  // CHECK: [[BARRIER:%.*]] = xw.barrier dep [[READS]]
  // CHECK: xw.store {{.*}} dep [[BARRIER]]
  llvm.func spir_kernelcc @atomic_and_barrier(%a: !llvm.ptr<1>,
                                             %b: !llvm.ptr<1>, %index: i64) {
    %one = llvm.mlir.constant(1 : i32) : i32
    %old = llvm.call spir_funccc @_Z10atomic_addPU3AS1Vjj(%a, %one)
        : (!llvm.ptr<1>, i32) -> i32
    %pa = llvm.getelementptr %a[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %first = llvm.load %pa : !llvm.ptr<1> -> i32
    %pb = llvm.getelementptr %b[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %second = llvm.load %pb : !llvm.ptr<1> -> i32
    %flags = llvm.mlir.constant(1 : i32) : i32
    llvm.call spir_funccc @_Z7barrierj(%flags) : (i32) -> ()
    llvm.store %first, %pa : i32, !llvm.ptr<1>
    llvm.return
  }

  llvm.func spir_funccc @_Z10atomic_addPU3AS1Vjj(!llvm.ptr<1>, i32) -> i32
  llvm.func spir_funccc @_Z7barrierj(i32)
}
