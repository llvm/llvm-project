// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// A three-element vector is classified at its width rounded up to a power of
// two.

// CIRGen cannot yet load or store a three-element vector, so plain vectors
// only flow from a call result into a return or another call.

typedef char c3 __attribute__((ext_vector_type(3)));
typedef short s3 __attribute__((ext_vector_type(3)));
typedef float f3 __attribute__((ext_vector_type(3)));
typedef double d3 __attribute__((ext_vector_type(3)));

c3 src_c3(void);
void sink_c3(c3);
s3 src_s3(void);
void sink_s3(s3);
f3 src_f3(void);
void sink_f3(f3);

// Three chars take 4 bytes.
c3 fwd_c3(void) { return src_c3(); }

// CIR: cir.func {{.*}}@fwd_c3() -> !u32i
// CIR: cir.call @src_c3() : () -> !u32i
// LLVM: define{{.*}} i32 @fwd_c3()
// LLVM: call i32 @src_c3()

void pass_c3(void) { sink_c3(src_c3()); }

// CIR: cir.func {{.*}}@pass_c3()
// CIR: cir.call @src_c3() : () -> !u32i
// CIR: cir.call @sink_c3(%{{.+}}) : (!u32i) -> ()
// LLVM: define{{.*}} void @pass_c3()
// LLVM: call i32 @src_c3()
// LLVM: call void @sink_c3(i32 %{{.+}})

// Three shorts take 8 bytes.
s3 fwd_s3(void) { return src_s3(); }

// CIR: cir.func {{.*}}@fwd_s3() -> !cir.double
// CIR: cir.call @src_s3() : () -> !cir.double
// LLVM: define{{.*}} double @fwd_s3()
// LLVM: call double @src_s3()

void pass_s3(void) { sink_s3(src_s3()); }

// CIR: cir.func {{.*}}@pass_s3()
// CIR: cir.call @src_s3() : () -> !cir.double
// CIR: cir.call @sink_s3(%{{.+}}) : (!cir.double) -> ()
// LLVM: define{{.*}} void @pass_s3()
// LLVM: call double @src_s3()
// LLVM: call void @sink_s3(double %{{.+}})

// Three floats take 16 bytes and keep their vector type.
f3 fwd_f3(void) { return src_f3(); }

// CIR: cir.func {{.*}}@fwd_f3() -> !cir.vector<3 x !cir.float>
// CIR: cir.call @src_f3() : () -> !cir.vector<3 x !cir.float>
// LLVM: define{{.*}} <3 x float> @fwd_f3()
// LLVM: call <3 x float> @src_f3()

void pass_f3(void) { sink_f3(src_f3()); }

// CIR: cir.func {{.*}}@pass_f3()
// CIR: cir.call @sink_f3(%{{.+}}) : (!cir.vector<3 x !cir.float> {llvm.noundef}) -> ()
// LLVM: define{{.*}} void @pass_f3()
// LLVM: call void @sink_f3(<3 x float> noundef %{{.+}})

struct SF3 { f3 v; };
struct SF3 sf3(struct SF3 s) { return s; }

// CIR: cir.func {{.*}}@sf3(%arg0: !cir.vector<3 x !cir.float> loc({{.+}})) -> !cir.vector<3 x !cir.float>
// LLVM: define{{.*}} <3 x float> @sf3(<3 x float> %{{.+}})

// Three doubles take 32 bytes, which goes to memory without AVX.
struct SD3 { d3 v; };
struct SD3 sd3(struct SD3 s) { return s; }

// CIR: cir.func {{.*}}@sd3(%arg0: !cir.ptr<!rec_SD3> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_SD3, llvm.writable} loc({{.+}}), %arg1: !cir.ptr<!rec_SD3> {llvm.align = 32 : i64, llvm.byval = !rec_SD3, llvm.noundef} loc(
// LLVM: define{{.*}} void @sd3(ptr dead_on_unwind noalias writable sret(%struct.SD3) align 32 %{{.+}}, ptr noundef byval(%struct.SD3) align 32 %{{.+}})
