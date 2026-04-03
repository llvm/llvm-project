// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

// CIR-DAG: !rec_Param2 = !cir.record<struct "Param2" incomplete>
// CIR-DAG: !rec_Ret1 = !cir.record<struct "Ret1" incomplete>
// CIR-DAG: !rec_S1 = !cir.record<struct "S1" {!cir.vptr}>
// CIR-DAG: !rec_S2 = !cir.record<struct "S2" {!cir.vptr}>
// LLVMCIR-DAG: %struct.Ret1 = type {}
// LLVMCIR-DAG: %struct.Param2 = type {}

struct Ret1;

struct S1 {
  virtual void key();
  virtual Ret1 badReturn();
};

void S1::key() {}

// CIR:  @_ZTV2S1 = #cir.vtable<{
// CIR-SAME: #cir.const_array<[
// CIR-SAME:   #cir.ptr<null> : !cir.ptr<!u8i>, 
// CIR-SAME:   #cir.global_view<@_ZTI2S1> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZN2S13keyEv> : !cir.ptr<!u8i>,
// CIR-SAME:   #cir.global_view<@_ZN2S19badReturnEv> : !cir.ptr<!u8i>]> :
// CIR-SAME: !cir.array<!cir.ptr<!u8i> x 4>}>

// LLVM: @_ZTV2S1 = {{.*}}{ [4 x ptr] } { [4 x ptr] [
// LLVM-SAME: ptr null, 
// LLVM-SAME: ptr @_ZTI2S1,
// LLVM-SAME: ptr @_ZN2S13keyEv,
// LLVM-SAME: ptr @_ZN2S19badReturnEv] }

struct Param2;

struct S2 {
  virtual void key();
  virtual void badParam(Param2 x);
};

void S2::key() {}
// CIR: @_ZTV2S2 = #cir.vtable<{
// CIR-SAME: #cir.const_array<[
// CIR-SAME:    #cir.ptr<null> : !cir.ptr<!u8i>,
// CIR-SAME:    #cir.global_view<@_ZTI2S2> : !cir.ptr<!u8i>,
// CIR-SAME:    #cir.global_view<@_ZN2S23keyEv> : !cir.ptr<!u8i>,
// CIR-SAME:    #cir.global_view<@_ZN2S28badParamE6Param2> : !cir.ptr<!u8i>]> :
// CIR-SAME: !cir.array<!cir.ptr<!u8i> x 4>}>

// LLVM: @_ZTV2S2 = {{.*}}{ [4 x ptr] } { [4 x ptr] [
// LLVM-SAME: ptr null, 
// LLVM-SAME: ptr @_ZTI2S2,
// LLVM-SAME: ptr @_ZN2S23keyEv,
// LLVM-SAME: ptr @_ZN2S28badParamE6Param2] }


// CIR: cir.func private @_ZN2S19badReturnEv(!cir.ptr<!rec_S1> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) -> !rec_Ret1
// LLVMCIR: declare %struct.Ret1 @_ZN2S19badReturnEv(ptr noundef nonnull align 8 dereferenceable(8))
// OGCG: declare void @_ZN2S19badReturnEv()
// CIR: cir.func private @_ZN2S28badParamE6Param2(!cir.ptr<!rec_S2> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !rec_Param2)
// LLVMCIR: declare void @_ZN2S28badParamE6Param2(ptr noundef nonnull align 8 dereferenceable(8), %struct.Param2)
// OGCG: declare void @_ZN2S28badParamE6Param2() unnamed_addr
