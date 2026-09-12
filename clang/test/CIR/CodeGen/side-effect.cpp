// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct Big { long a, b, c, d; };
struct Small { int x; };

extern "C" {

__attribute__((const)) Big const_sret();
__attribute__((pure)) Big pure_sret();
__attribute__((const)) int const_byval(Big b);
__attribute__((pure)) int pure_byval(Big b);
__attribute__((const)) Small const_small();
__attribute__((const)) int const_ptr(const int *p);
__attribute__((const)) int const_variadic_decl(int n, ...);
__attribute__((const)) int const_byval2(int n, Big b);
__attribute__((const)) int const_bigref(const Big &b);

// FIXME: We should figure out how to better print this on functions in the
// future.
// CIR: cir.func{{.*}}@pure_func() -> !s32i side_effect(pure) attributes {{{.*}}nothrow} {
__attribute__((pure))
int pure_func() { return 2;}

// CIR: cir.func{{.*}}@const_func() -> !s32i side_effect(const) attributes {{{.*}}nothrow} {
__attribute__((const))
int const_func() { return 1;}

// Variadic definition: widened with no indirect slot in the signature.
// CIR: cir.func{{.*}}@const_variadic(%arg0: !s32i {{.*}}, ...) -> !s32i side_effect(const)
__attribute__((const))
int const_variadic(int n, ...) { return n; }

// A definition also gets llvm.noalias on the sret slot, so this is where
// noalias, writable and the widened effect have to coexist.
// CIR: cir.func{{.*}}@const_sret_def(%arg0: !cir.ptr<!rec_Big> {{{.*}}llvm.noalias{{.*}}llvm.sret = !rec_Big{{.*}}llvm.writable{{.*}}) side_effect(const)
__attribute__((const))
Big const_sret_def() { Big r{}; return r; }

void use() {
  // Unwidened at a call site: neither takes an indirect slot.
  // CIR: cir.call @pure_func() side_effect(pure)
  pure_func();
  // CIR: cir.call @const_func() side_effect(const)
  const_func();

  // The pass has already given these calls their sret operand.
  // CIR: cir.call @const_sret(%{{.+}}) side_effect(const)
  const_sret();
  // CIR: cir.call @pure_sret(%{{.+}}) side_effect(pure)
  pure_sret();

  Big b{};
  int i = 0;
  // CIR: cir.call @const_byval({{.*}}) side_effect(const)
  const_byval(b);
  // CIR: cir.call @pure_byval({{.*}}) side_effect(pure)
  pure_byval(b);

  // CIR: cir.call @const_small() side_effect(const)
  const_small();
  // A source-level pointer is not ABI-introduced memory, so the effect stays
  // memory(none).
  // CIR: cir.call @const_ptr({{.*}}) side_effect(const)
  const_ptr(&i);

  // A call site sees the arguments it passes, so it is not widened for being
  // variadic.
  // CIR: cir.call @const_variadic({{.*}}) side_effect(const)
  const_variadic(1);
  // CIR: cir.call @const_variadic_decl({{.*}}) side_effect(const)
  const_variadic_decl(1);

  // CIR: cir.call @const_byval2({{.*}}) side_effect(const) : (!s32i{{.*}}, !cir.ptr<!rec_Big> {{{.*}}llvm.byval{{.*}}}) -> !s32i
  const_byval2(1, b);
  // CIR: cir.call @const_sret_def(%{{.+}}) side_effect(const)
  const_sret_def();

  // A reference is a direct pointer slot that still carries align, so neither
  // align nor a record pointee can stand in for an indirect slot.
  // CIR: cir.call @const_bigref({{.*}}) side_effect(const)
  const_bigref(b);
}

}

// The named functions appear in the same relative order in both emits, so
// these checks are ordered.  The intrinsic declarations do not, so each
// attribute group is captured off its own define or declare line instead of
// matched adjacent to it.

// Definitions.
// LLVM: define{{.*}} i32 @pure_func() #[[READ_DEF:[0-9]+]] {
// LLVM: define{{.*}} i32 @const_func() #[[NONE_DEF:[0-9]+]] {
// LLVM: define{{.*}} i32 @const_variadic(i32 noundef %{{[^,)]+}}, ...) #[[ARGMEM_DEF:[0-9]+]] {
// LLVM: define{{.*}} void @const_sret_def(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{[^,)]+}}) #[[ARGMEM_DEF]] {

// Call sites.
// LLVM: define{{.*}} void @use()
// LLVM: call i32 @pure_func() #[[READ_CALL:[0-9]+]]
// LLVM: call i32 @const_func() #[[NONE_CALL:[0-9]+]]
// LLVM: call void @const_sret(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{.+}}) #[[ARGMEM_CALL:[0-9]+]]
// LLVM: call void @pure_sret(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{.+}}) #[[READ_ARGMEM_CALL:[0-9]+]]
// LLVM: call i32 @const_byval(ptr noundef byval(%struct.Big) align 8 %{{.+}}) #[[ARGMEM_CALL]]
// LLVM: call i32 @pure_byval(ptr noundef byval(%struct.Big) align 8 %{{.+}}) #[[READ_ARGMEM_CALL]]
// LLVM: call i32 @const_small() #[[NONE_CALL]]
// LLVM: call i32 @const_ptr(ptr noundef %{{.+}}) #[[NONE_CALL]]
// LLVM: call i32 (i32, ...) @const_variadic(i32 noundef 1) #[[NONE_CALL]]
// LLVM: call i32 (i32, ...) @const_variadic_decl(i32 noundef 1) #[[NONE_CALL]]
// LLVM: call i32 @const_byval2(i32 noundef 1, ptr noundef byval(%struct.Big) align 8 %{{.+}}) #[[ARGMEM_CALL]]
// LLVM: call void @const_sret_def(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{.+}}) #[[ARGMEM_CALL]]
// LLVM: call i32 @const_bigref(ptr noundef nonnull align 8 dereferenceable(32) %{{.+}}) #[[NONE_CALL]]

// Declarations.
// LLVM: declare void @const_sret(ptr dead_on_unwind writable sret(%struct.Big) align 8) #[[ARGMEM_DECL:[0-9]+]]
// LLVM: declare void @pure_sret(ptr dead_on_unwind writable sret(%struct.Big) align 8) #[[READ_ARGMEM_DECL:[0-9]+]]
// LLVM: declare i32 @const_byval(ptr noundef byval(%struct.Big) align 8) #[[ARGMEM_DECL]]
// LLVM: declare i32 @pure_byval(ptr noundef byval(%struct.Big) align 8) #[[READ_ARGMEM_DECL]]
// LLVM: declare i32 @const_small() #[[NONE_DECL:[0-9]+]]
// LLVM: declare i32 @const_ptr(ptr noundef) #[[NONE_DECL]]
// LLVM: declare i32 @const_variadic_decl(i32 noundef, ...) #[[ARGMEM_DECL]]
// LLVM: declare i32 @const_byval2(i32 noundef, ptr noundef byval(%struct.Big) align 8) #[[ARGMEM_DECL]]
// LLVM: declare i32 @const_bigref(ptr noundef nonnull align 8 dereferenceable(32)) #[[NONE_DECL]]

// The trailing wildcard covers target-features and the other codegen-option
// strings, which differ between the emits.
// LLVM-DAG: attributes #[[READ_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(read) {{.*}}}
// LLVM-DAG: attributes #[[NONE_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(none) {{.*}}}
// LLVM-DAG: attributes #[[ARGMEM_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[ARGMEM_DECL]] = { nounwind willreturn memory(argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[READ_ARGMEM_DECL]] = { nounwind willreturn memory(read, argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[NONE_DECL]] = { nounwind willreturn memory(none) {{.*}}}
// LLVM-DAG: attributes #[[READ_CALL]] = { nounwind willreturn memory(read) }
// LLVM-DAG: attributes #[[NONE_CALL]] = { nounwind willreturn memory(none) }
// LLVM-DAG: attributes #[[ARGMEM_CALL]] = { nounwind willreturn memory(argmem: readwrite) }
// LLVM-DAG: attributes #[[READ_ARGMEM_CALL]] = { nounwind willreturn memory(read, argmem: readwrite) }
