// CIR calling-convention lowering does not handle SVE scalable vector types
// yet, so the -fclangir runs disable it.
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -disable-O0-optnone -fclangir -fno-clangir-call-conv-lowering -fdefined-int-divrem -emit-cir -o - %s | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -disable-O0-optnone -fclangir -fno-clangir-call-conv-lowering -fdefined-int-divrem -emit-llvm -o - %s | opt -S -passes=sroa | FileCheck --check-prefix=LLVM %s

typedef __SVInt32_t svint32_t;
typedef __SVUint32_t svuint32_t;
typedef svint32_t fixed_svint32_t __attribute__((arm_sve_vector_bits(128)));

// CIR-LABEL: cir.func {{.*}} @sve_sdiv(
// CIR:       cir.vec.cmp(ne, {{.*}}) : !cir.vector<[4] x !s32i, sve_data>, !cir.vector<[4] x !s32i, sve_data>
// CIR:       cir.vec.ternary({{.*}}) : !cir.vector<[4] x !s32i, sve_data>, !cir.vector<[4] x !s32i, sve_data>
// CIR:       cir.div {{.*}} : !cir.vector<[4] x !s32i, sve_data>
// LLVM-LABEL: define {{.*}} <vscale x 4 x i32> @sve_sdiv(
// LLVM:       icmp ne <vscale x 4 x i32> %{{.*}}, splat (i32 -1)
// LLVM:       select <vscale x 4 x i1>
// LLVM:       icmp ne <vscale x 4 x i32> %{{.*}}, zeroinitializer
// LLVM:       select <vscale x 4 x i1>
// LLVM:       sdiv <vscale x 4 x i32>
svint32_t sve_sdiv(svint32_t a, svint32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @sve_urem(
// CIR:       cir.vec.cmp(ne, {{.*}}) : !cir.vector<[4] x !u32i, sve_data>, !cir.vector<[4] x !s32i, sve_data>
// CIR:       cir.vec.ternary({{.*}}) : !cir.vector<[4] x !s32i, sve_data>, !cir.vector<[4] x !u32i, sve_data>
// CIR:       cir.rem {{.*}} : !cir.vector<[4] x !u32i, sve_data>
// LLVM-LABEL: define {{.*}} <vscale x 4 x i32> @sve_urem(
// LLVM:       icmp ne <vscale x 4 x i32> %{{.*}}, zeroinitializer
// LLVM:       select <vscale x 4 x i1> {{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> splat (i32 1)
// LLVM:       urem <vscale x 4 x i32>
svuint32_t sve_urem(svuint32_t a, svuint32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @sve_fixed_srem(
// CIR:       cir.vec.cmp(ne, {{.*}}) : !cir.vector<4 x !s32i, sve_data>, !cir.vector<4 x !s32i, sve_data>
// CIR:       cir.rem {{.*}} : !cir.vector<4 x !s32i, sve_data>
// LLVM-LABEL: define {{.*}} <4 x i32> @sve_fixed_srem(
// LLVM:       srem <4 x i32>
fixed_svint32_t sve_fixed_srem(fixed_svint32_t a, fixed_svint32_t b) {
  return a % b;
}
