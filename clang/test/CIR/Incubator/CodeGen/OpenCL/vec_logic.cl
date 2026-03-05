// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-cir -fclangir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-llvm -fclangir -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

kernel void test(char4 in1, char4 in2, local char4 *out)
{
    *out = (in1 == (char4)3 && (in1 == (char4)5 || in2 == (char4)7))
            ? in1 : in2;
}


// CIR: [[ZERO:%.*]] = cir.const #cir.zero : !cir.vector<!s8i x 4>
// CIR: [[CMP1:%.*]] = cir.vec.cmp(ne, %{{.*}}, [[ZERO]]) : !cir.vector<!s8i x 4>, !cir.vector<!cir.bool x 4>
// CIR: [[CMP2:%.*]] = cir.vec.cmp(ne, %{{.*}}, [[ZERO]]) : !cir.vector<!s8i x 4>, !cir.vector<!cir.bool x 4>
// CIR: [[OR:%.*]] = cir.binop(or, [[CMP1]], [[CMP2]]) : !cir.vector<!cir.bool x 4>
// CIR: [[CAST1:%.*]] = cir.cast bool_to_int [[OR]] : !cir.vector<!cir.bool x 4> -> !cir.vector<!s8i x 4>
// CIR: [[ZERO2:%.*]] = cir.const #cir.zero : !cir.vector<!s8i x 4>
// CIR: [[CMP3:%.*]] = cir.vec.cmp(ne, %{{.*}}, [[ZERO2]]) : !cir.vector<!s8i x 4>, !cir.vector<!cir.bool x 4>
// CIR: [[CMP4:%.*]] = cir.vec.cmp(ne, [[CAST1]], [[ZERO2]]) : !cir.vector<!s8i x 4>, !cir.vector<!cir.bool x 4>
// CIR: [[AND:%.*]] = cir.binop(and, [[CMP3]], [[CMP4]]) : !cir.vector<!cir.bool x 4>
// CIR: cir.cast bool_to_int [[AND]] : !cir.vector<!cir.bool x 4> -> !cir.vector<!s8i x 4>

// LLVM: [[CMP1:%.*]] = icmp ne <4 x i8> %{{.*}}, zeroinitializer
// LLVM: [[CMP2:%.*]] = icmp ne <4 x i8> %{{.*}}, zeroinitializer
// LLVM: [[OR:%.*]] = or <4 x i1> [[CMP1]], [[CMP2]]
// LLVM: [[SEXT:%.*]] = sext <4 x i1> [[OR]] to <4 x i8>
// LLVM: [[CMP3:%.*]] = icmp ne <4 x i8> %{{.*}}, zeroinitializer
// LLVM: [[CMP4:%.*]] = icmp ne <4 x i8> [[SEXT]], zeroinitializer
// LLVM: [[AND:%.*]] = and <4 x i1> [[CMP3]], [[CMP4]]
// LLVM: sext <4 x i1> [[AND]] to <4 x i8>

// OG-LLVM: [[CMP1:%.*]] = icmp ne <4 x i8> %{{.*}}, zeroinitializer
// OG-LLVM: [[CMP2:%.*]] = icmp ne <4 x i8> %{{.*}}, zeroinitializer
// OG-LLVM: [[OR:%.*]] = or <4 x i1> [[CMP1]], [[CMP2]]
// OG-LLVM: [[SEXT:%.*]] = sext <4 x i1> [[OR]] to <4 x i8>
// OG-LLVM: [[CMP3:%.*]] = icmp ne <4 x i8> %{{.*}}, zeroinitializer
// OG-LLVM: [[CMP4:%.*]] = icmp ne <4 x i8> [[SEXT]], zeroinitializer
// OG-LLVM: [[AND:%.*]] = and <4 x i1> [[CMP3]], [[CMP4]]
// OG-LLVM: sext <4 x i1> [[AND]] to <4 x i8>