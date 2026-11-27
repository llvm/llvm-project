// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -target-feature +sve -mvscale-min=1 -mvscale-max=1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -target-feature +sve -mvscale-min=1 -mvscale-max=1 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -target-feature +sve -mvscale-min=1 -mvscale-max=1 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

// __mfp8 values are kept out of function signatures here because calling
// convention lowering does not yet handle the type.

__mfp8 g;
__mfp8 garr[4];
__Mfloat8x8_t gv8;
__Mfloat8x16_t gv16;

struct S {
  __mfp8 x;
};

struct S s;

typedef __SVMfloat8_t fixed_mf8_t __attribute__((arm_sve_vector_bits(128)));
fixed_mf8_t gfixed;

// CIR: !rec_S = !cir.struct<"S" {data !cir.mfp8}>
// CIR: cir.global external @g = #cir.zero : !cir.mfp8 align(1)
// CIR: cir.global external @gv16 = #cir.zero : !cir.vector<16 x !cir.mfp8> align(16)
// CIR: cir.global external @gv8 = #cir.zero : !cir.vector<8 x !cir.mfp8> align(8)
// CIR: cir.global external @s = #cir.zero : !rec_S align(1)
// CIR: cir.global external @garr = #cir.zero : !cir.array<!cir.mfp8 x 4> align(1)
// CIR: cir.global external @gfixed = #cir.zero : !cir.vector<16 x !cir.mfp8, sve_data> align(16)

// LLVM: %struct.S = type { <1 x i8> }
// LLVM: @g = global <1 x i8> zeroinitializer, align 1
// LLVM: @gv16 = global <16 x i8> zeroinitializer, align 16
// LLVM: @gv8 = global <8 x i8> zeroinitializer, align 8
// LLVM: @s = global %struct.S zeroinitializer, align 1
// LLVM: @garr = global [4 x <1 x i8>] zeroinitializer, align 1
// LLVM: @gfixed = global <16 x i8> zeroinitializer, align 16

void copy_local(void) {
  __mfp8 a[10];
  a[2] = g;
  g = a[2];
}

// CIR-LABEL: cir.func {{.*}} @copy_local()
// CIR:   %[[A:.*]] = cir.alloca "a" align(1) : !cir.ptr<!cir.array<!cir.mfp8 x 10>>
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!cir.mfp8>
// CIR:   %[[G_VAL:.*]] = cir.load align(1) %[[G]] : !cir.ptr<!cir.mfp8>, !cir.mfp8
// CIR:   %[[ELT:.*]] = cir.get_element inbounds %[[A]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.mfp8 x 10>> -> !cir.ptr<!cir.mfp8>
// CIR:   cir.store align(1) %[[G_VAL]], %[[ELT]] : !cir.mfp8, !cir.ptr<!cir.mfp8>
// CIR:   %[[ELT2:.*]] = cir.get_element inbounds %[[A]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.mfp8 x 10>> -> !cir.ptr<!cir.mfp8>
// CIR:   %[[ELT_VAL:.*]] = cir.load align(1) %[[ELT2]] : !cir.ptr<!cir.mfp8>, !cir.mfp8
// CIR:   %[[G2:.*]] = cir.get_global @g : !cir.ptr<!cir.mfp8>
// CIR:   cir.store align(1) %[[ELT_VAL]], %[[G2]] : !cir.mfp8, !cir.ptr<!cir.mfp8>

// LLVM-LABEL: define dso_local void @copy_local()
// LLVM:   %[[A:.*]] = alloca [10 x <1 x i8>], align 1
// LLVM:   %[[G_VAL:.*]] = load <1 x i8>, ptr @g, align 1
// LLVM:   %[[ELT:.*]] = getelementptr inbounds [10 x <1 x i8>], ptr %[[A]], i64 0, i64 2
// LLVM:   store <1 x i8> %[[G_VAL]], ptr %[[ELT]], align 1
// LLVM:   %[[ELT2:.*]] = getelementptr inbounds [10 x <1 x i8>], ptr %[[A]], i64 0, i64 2
// LLVM:   %[[ELT_VAL:.*]] = load <1 x i8>, ptr %[[ELT2]], align 1
// LLVM:   store <1 x i8> %[[ELT_VAL]], ptr @g, align 1

void extract_element(int i) {
  g = gv16[i];
}

// CIR-LABEL: cir.func {{.*}} @extract_element(
// CIR:   %[[GV16:.*]] = cir.get_global @gv16 : !cir.ptr<!cir.vector<16 x !cir.mfp8>>
// CIR:   %[[VEC:.*]] = cir.load align(16) %[[GV16]] : !cir.ptr<!cir.vector<16 x !cir.mfp8>>, !cir.vector<16 x !cir.mfp8>
// CIR:   %[[I:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   %[[EXT:.*]] = cir.vec.extract %[[VEC]][%[[I]] : !s32i] : !cir.vector<16 x !cir.mfp8>
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!cir.mfp8>
// CIR:   cir.store align(1) %[[EXT]], %[[G]] : !cir.mfp8, !cir.ptr<!cir.mfp8>

// LLVM-LABEL: define dso_local void @extract_element(
// LLVM:   %[[VEC:.*]] = load <16 x i8>, ptr @gv16, align 16
// LLVM:   %[[I:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:   %[[EXT:.*]] = extractelement <16 x i8> %[[VEC]], i32 %[[I]]
// LLVM:   %[[CAST:.*]] = insertelement <1 x i8> poison, i8 %[[EXT]], i64 0
// LLVM:   store <1 x i8> %[[CAST]], ptr @g, align 1

void insert_element(int i) {
  gv16[i] = g;
}

// CIR-LABEL: cir.func {{.*}} @insert_element(
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!cir.mfp8>
// CIR:   %[[G_VAL:.*]] = cir.load align(1) %[[G]] : !cir.ptr<!cir.mfp8>, !cir.mfp8
// CIR:   %[[GV16:.*]] = cir.get_global @gv16 : !cir.ptr<!cir.vector<16 x !cir.mfp8>>
// CIR:   %[[I:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:   %[[VEC:.*]] = cir.load align(16) %[[GV16]] : !cir.ptr<!cir.vector<16 x !cir.mfp8>>, !cir.vector<16 x !cir.mfp8>
// CIR:   %[[INS:.*]] = cir.vec.insert %[[G_VAL]], %[[VEC]][%[[I]] : !s32i] : !cir.vector<16 x !cir.mfp8>
// CIR:   cir.store align(16) %[[INS]], %[[GV16]] : !cir.vector<16 x !cir.mfp8>, !cir.ptr<!cir.vector<16 x !cir.mfp8>>

// LLVM-LABEL: define dso_local void @insert_element(
// LLVM:   %[[G_VAL:.*]] = load <1 x i8>, ptr @g, align 1
// LLVM:   %[[I:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM:   %[[VEC:.*]] = load <16 x i8>, ptr @gv16, align 16
// LLVM:   %[[ELT:.*]] = bitcast <1 x i8> %[[G_VAL]] to i8
// LLVM:   %[[INS:.*]] = insertelement <16 x i8> %[[VEC]], i8 %[[ELT]], i32 %[[I]]
// LLVM:   store <16 x i8> %[[INS]], ptr @gv16, align 16

void vector_init(void) {
  gv8 = (__Mfloat8x8_t){g, g, g, g, g, g, g, g};
}

// CIR-LABEL: cir.func {{.*}} @vector_init()
// CIR:   %[[CL:.*]] = cir.alloca ".compoundliteral" align(8) : !cir.ptr<!cir.vector<8 x !cir.mfp8>>
// CIR:   %[[VEC:.*]] = cir.vec.create(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !cir.mfp8, !cir.mfp8, !cir.mfp8, !cir.mfp8, !cir.mfp8, !cir.mfp8, !cir.mfp8, !cir.mfp8) : !cir.vector<8 x !cir.mfp8>
// CIR:   cir.store align(8) %[[VEC]], %[[CL]] : !cir.vector<8 x !cir.mfp8>, !cir.ptr<!cir.vector<8 x !cir.mfp8>>
// CIR:   %[[VAL:.*]] = cir.load align(8) %[[CL]] : !cir.ptr<!cir.vector<8 x !cir.mfp8>>, !cir.vector<8 x !cir.mfp8>
// CIR:   %[[GV8:.*]] = cir.get_global @gv8 : !cir.ptr<!cir.vector<8 x !cir.mfp8>>
// CIR:   cir.store align(8) %[[VAL]], %[[GV8]] : !cir.vector<8 x !cir.mfp8>, !cir.ptr<!cir.vector<8 x !cir.mfp8>>

// LLVM-LABEL: define dso_local void @vector_init()
// Classic codegen builds the vector by shuffling in each <1 x i8> element.
// CIR bitcasts each element to i8 and inserts it.
// LLVM:      %[[CL:.*]] = alloca <8 x i8>, align 8
// LLVM:      %[[E0:.*]] = load <1 x i8>, ptr @g, align 1
// LLVMCIR:   %[[E0_I8:.*]] = bitcast <1 x i8> %[[E0]] to i8
// LLVMCIR:   %{{.*}} = insertelement <8 x i8> poison, i8 %[[E0_I8]], i64 0
// LLVMCIR:   %[[VEC:.*]] = insertelement <8 x i8> %{{.*}}, i8 %{{.*}}, i64 7
// OGCG:      %[[VEC:.*]] = shufflevector <8 x i8> %{{.*}}, <8 x i8> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
// LLVM:      store <8 x i8> %[[VEC]], ptr %[[CL]], align 8
// LLVM:      %[[VAL:.*]] = load <8 x i8>, ptr %[[CL]], align 8
// LLVM:      store <8 x i8> %[[VAL]], ptr @gv8, align 8

void struct_init(void) {
  s = (struct S){g};
}

// CIR-LABEL: cir.func {{.*}} @struct_init()
// CIR:   %[[CL:.*]] = cir.alloca ".compoundliteral" align(1) : !cir.ptr<!rec_S>
// CIR:   %[[S:.*]] = cir.get_global @s : !cir.ptr<!rec_S>
// CIR:   %[[X:.*]] = cir.get_member %[[CL]][0] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!cir.mfp8>
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!cir.mfp8>
// CIR:   %[[G_VAL:.*]] = cir.load align(1) %[[G]] : !cir.ptr<!cir.mfp8>, !cir.mfp8
// CIR:   cir.store align(1) %[[G_VAL]], %[[X]] : !cir.mfp8, !cir.ptr<!cir.mfp8>
// CIR:   cir.copy %[[CL]] align(1) to %[[S]] align(1) : !cir.ptr<!rec_S>

// LLVM-LABEL: define dso_local void @struct_init()
// LLVM:      %[[CL:.*]] = alloca %struct.S, align 1
// LLVM:      %[[X:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[CL]], i32 0, i32 0
// LLVM:      %[[G_VAL:.*]] = load <1 x i8>, ptr @g, align 1
// LLVM:      store <1 x i8> %[[G_VAL]], ptr %[[X]], align 1
// LLVM:      call void @llvm.memcpy.p0.p0.i64(ptr align 1 @s, ptr align 1 %[[CL]], i64 1, i1 false)

void sve_local(void) {
  __SVMfloat8_t a, b;
  a = b;
}

// CIR-LABEL: cir.func {{.*}} @sve_local()
// CIR:   %[[A:.*]] = cir.alloca "a" align(16) : !cir.ptr<!cir.vector<[16] x !cir.mfp8, sve_data>>
// CIR:   %[[B:.*]] = cir.alloca "b" align(16) : !cir.ptr<!cir.vector<[16] x !cir.mfp8, sve_data>>
// CIR:   %[[B_VAL:.*]] = cir.load align(16) %[[B]] : !cir.ptr<!cir.vector<[16] x !cir.mfp8, sve_data>>, !cir.vector<[16] x !cir.mfp8, sve_data>
// CIR:   cir.store align(16) %[[B_VAL]], %[[A]] : !cir.vector<[16] x !cir.mfp8, sve_data>, !cir.ptr<!cir.vector<[16] x !cir.mfp8, sve_data>>

// LLVM-LABEL: define dso_local void @sve_local()
// LLVM:   %[[A:.*]] = alloca <vscale x 16 x i8>, align 16
// LLVM:   %[[B:.*]] = alloca <vscale x 16 x i8>, align 16
// LLVM:   %[[B_VAL:.*]] = load <vscale x 16 x i8>, ptr %[[B]], align 16
// LLVM:   store <vscale x 16 x i8> %[[B_VAL]], ptr %[[A]], align 16
