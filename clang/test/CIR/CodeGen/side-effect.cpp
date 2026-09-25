// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclspec -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclspec -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclspec -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct Big { long a, b, c, d; };
struct Small { int x; };
struct WithDtor { int x; ~WithDtor(); };
struct Empty {};
struct SmallPair { int a, b; };
struct TwoLong { long a, b; };

extern "C" {

__attribute__((const)) Big const_sret();
__attribute__((pure)) Big pure_sret();
__attribute__((const)) int const_byval(Big b);
__attribute__((pure)) int pure_byval(Big b);
__attribute__((const)) int const_byval2(int n, Big b);
__attribute__((const)) int const_non_byval(WithDtor w);
__attribute__((const)) int const_variadic_decl(int n, ...);
__attribute__((const)) Small const_small();
__attribute__((const)) int const_ptr(const int *p);
__attribute__((const)) int const_bigref(const Big &b);
__attribute__((const)) int const_ignore(Empty e);
__attribute__((const)) int const_coerced(SmallPair p);
__attribute__((const)) int const_flattened(TwoLong t);

// FIXME: We should figure out how to better print this on functions in the
// future.
// CIR: cir.func{{.*}}@pure_func() -> !s32i attributes {{{.*}}nothrow, nounwind, willreturn, memory_effects = #cir.memory_effects<other = read, arg_mem = read, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} {
__attribute__((pure))
int pure_func() { return 2;}

// CIR: cir.func{{.*}}@const_func() -> !s32i attributes {{{.*}}nothrow, nounwind, willreturn, memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} {
__attribute__((const))
int const_func() { return 1;}

// CIR: cir.func{{.*}}@noalias_func(%{{[^,)]+}}: !cir.ptr<!s32i> {llvm.noundef}{{.*}}) -> !s32i attributes {{{.*}}nothrow, nounwind, memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = readwrite, errno_mem = none, target_mem0 = none, target_mem1 = none>} {
__declspec(noalias)
int noalias_func(int *p) { return *p; }

// Widened with no indirect slot in the signature: an ellipsis argument can be
// a pointer the declared parameters say nothing about.
// CIR: cir.func{{.*}}@const_variadic(%{{[^,)]+}}: !s32i {llvm.noundef}{{.*}}, ...) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} {
__attribute__((const))
int const_variadic(int n, ...) { return n; }

// A definition also gets llvm.noalias on the sret slot, so this is where
// noalias, writable and the widened effect have to coexist.
// CIR: cir.func{{.*}}@const_sret_def(%{{[^,)]+}}: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Big, llvm.writable}{{.*}}) attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} {
__attribute__((const))
Big const_sret_def() { Big r{}; return r; }

// CIR: cir.func{{.*}}@pure_sret_def(%{{[^,)]+}}: !cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}}{{.*}}) attributes {{{.*}}memory_effects = #cir.memory_effects<other = read, arg_mem = readwrite, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} {
__attribute__((pure))
Big pure_sret_def() { Big r{}; return r; }

void use() {
  // CIR: cir.call @pure_func() nounwind willreturn {memory_effects = #cir.memory_effects<other = read, arg_mem = read, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} : () -> !s32i
  pure_func();
  // CIR: cir.call @const_func() nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : () -> !s32i
  const_func();
  int x = 0;
  // CIR: cir.call @noalias_func(%{{[^,)]+}}) nounwind {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = readwrite, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!s32i> {llvm.noundef}) -> !s32i
  noalias_func(&x);

  // The pass has already given these calls their sret operand.
  // CIR: cir.call @const_sret(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}}) -> ()
  const_sret();
  // CIR: cir.call @pure_sret(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = read, arg_mem = readwrite, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} : (!cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}}) -> ()
  pure_sret();

  Big b{};
  // CIR: cir.call @const_byval(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!rec_Big> {{{.*}}llvm.byval = !rec_Big{{.*}}}) -> !s32i
  const_byval(b);
  // CIR: cir.call @pure_byval(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = read, arg_mem = readwrite, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} : (!cir.ptr<!rec_Big> {{{.*}}llvm.byval = !rec_Big{{.*}}}) -> !s32i
  pure_byval(b);

  // A record small enough to come back in a register hands the callee
  // nothing, so the return side leaves the effects alone.
  // CIR: cir.call @const_small() nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : () -> !s32i
  const_small();
  // A pointer the source wrote is not memory the ABI handed over.
  // CIR: cir.call @const_ptr(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!s32i> {llvm.noundef}) -> !s32i
  const_ptr(&x);

  // A call passes a fixed argument list, so neither of these widens even
  // though both callees are variadic and both were widened themselves.
  // CIR: cir.call @const_variadic(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!s32i {llvm.noundef}) -> !s32i
  const_variadic(1);
  // CIR: cir.call @const_variadic_decl(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!s32i {llvm.noundef}) -> !s32i
  const_variadic_decl(1);

  // A leading direct argument puts the byval slot at index 1.
  // CIR: cir.call @const_byval2(%{{[^,)]+}}, %{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!s32i {llvm.noundef}, !cir.ptr<!rec_Big> {{{.*}}llvm.byval = !rec_Big{{.*}}}) -> !s32i
  const_byval2(1, b);
  // CIR: cir.call @const_sret_def(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}}) -> ()
  const_sret_def();

  // A reference parameter is a pointer the source wrote, not a slot the ABI
  // introduced.
  // CIR: cir.call @const_bigref(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) -> !s32i
  const_bigref(b);

  // A class with a non-trivial destructor passes indirectly without byval,
  // and widens the same as byval does.
  WithDtor w{};
  // CIR: cir.call @const_non_byval(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!rec_WithDtor> {{{.*}}llvm.nofreeobj{{.*}}}) -> !s32i
  const_non_byval(w);

  // The remaining argument rewrites hand the callee no memory: the argument
  // is dropped, coerced into a register, or flattened into registers.
  // CIR: cir.call @const_ignore() nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : () -> !s32i
  const_ignore(Empty{});
  // CIR: cir.call @const_coerced(%{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!u64i) -> !s32i
  const_coerced(SmallPair{});
  // CIR: cir.call @const_flattened(%{{[^,)]+}}, %{{[^,)]+}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!s64i, !s64i) -> !s32i
  const_flattened(TwoLong{});
}

// The declarations print after @use(), in the order the calls above reach
// them.
// CIR: cir.func private @const_sret(!cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}}) attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @pure_sret(!cir.ptr<!rec_Big> {{{.*}}llvm.sret = !rec_Big{{.*}}}) attributes {{{.*}}memory_effects = #cir.memory_effects<other = read, arg_mem = readwrite, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>}
// CIR: cir.func private @const_byval(!cir.ptr<!rec_Big> {{{.*}}llvm.byval = !rec_Big{{.*}}}) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @pure_byval(!cir.ptr<!rec_Big> {{{.*}}llvm.byval = !rec_Big{{.*}}}) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = read, arg_mem = readwrite, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>}
// CIR: cir.func private @const_small() -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_ptr(!cir.ptr<!s32i> {llvm.noundef}) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_variadic_decl(!s32i {llvm.noundef}, ...) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_byval2(!s32i {llvm.noundef}, !cir.ptr<!rec_Big> {{{.*}}llvm.byval = !rec_Big{{.*}}}) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_bigref(!cir.ptr<!rec_Big> {{{.*}}llvm.nonnull{{.*}}}) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_non_byval(!cir.ptr<!rec_WithDtor> {{{.*}}llvm.nofreeobj{{.*}}}) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_ignore() -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_coerced(!u64i) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}
// CIR: cir.func private @const_flattened(!s64i, !s64i) -> !s32i attributes {{{.*}}memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>}

}

// The named functions appear in the same relative order in both emits, so
// these checks are ordered.  The intrinsic declarations do not, so each
// attribute group is captured off its own define or declare line rather than
// matched adjacent to it.

// Definitions.
// LLVM: define{{.*}} i32 @pure_func() #[[READ_DEF:[0-9]+]] {
// LLVM: define{{.*}} i32 @const_func() #[[NONE_DEF:[0-9]+]] {
// LLVM: define{{.*}} i32 @noalias_func(ptr noundef %{{[^,)]+}}) #[[NOALIAS_DEF:[0-9]+]] {
// LLVM: define{{.*}} i32 @const_variadic(i32 noundef %{{[^,)]+}}, ...) #[[ARGMEM_DEF:[0-9]+]] {
// LLVM: define{{.*}} void @const_sret_def(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{[^,)]+}}) #[[ARGMEM_DEF]] {
// LLVM: define{{.*}} void @pure_sret_def(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{[^,)]+}}) #[[READ_ARGMEM_DEF:[0-9]+]] {

// Call sites.
// LLVM: define{{.*}} void @use()
// LLVM: call i32 @pure_func() #[[READ_CALL:[0-9]+]]
// LLVM: call i32 @const_func() #[[NONE_CALL:[0-9]+]]
// LLVM: call i32 @noalias_func(ptr noundef %{{[^,)]+}}) #[[NOALIAS_CALL:[0-9]+]]
// LLVM: call void @const_sret(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{[^,)]+}}) #[[ARGMEM_CALL:[0-9]+]]
// LLVM: call void @pure_sret(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{[^,)]+}}) #[[READ_ARGMEM_CALL:[0-9]+]]
// LLVM: call i32 @const_byval(ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}}) #[[ARGMEM_CALL]]
// LLVM: call i32 @pure_byval(ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}}) #[[READ_ARGMEM_CALL]]
// LLVM: call i32 @const_small() #[[NONE_CALL]]
// LLVM: call i32 @const_ptr(ptr noundef %{{[^,)]+}}) #[[NONE_CALL]]
// LLVM: call i32 (i32, ...) @const_variadic(i32 noundef 1) #[[NONE_CALL]]
// LLVM: call i32 (i32, ...) @const_variadic_decl(i32 noundef 1) #[[NONE_CALL]]
// LLVM: call i32 @const_byval2(i32 noundef 1, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}}) #[[ARGMEM_CALL]]
// LLVM: call void @const_sret_def(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{[^,)]+}}) #[[ARGMEM_CALL]]
// LLVM: call i32 @const_bigref(ptr noundef nonnull align 8 dereferenceable(32) %{{[^,)]+}}) #[[NONE_CALL]]
// LLVM: call i32 @const_non_byval(ptr nofreeobj noundef align 4 dereferenceable(4) %{{[^,)]+}}) #[[ARGMEM_CALL]]
// LLVM: call i32 @const_ignore() #[[NONE_CALL]]
// LLVM: call i32 @const_coerced(i64 %{{[^,)]+}}) #[[NONE_CALL]]
// LLVM: call i32 @const_flattened(i64 %{{[^,)]+}}, i64 %{{[^,)]+}}) #[[NONE_CALL]]

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
// LLVM: declare i32 @const_non_byval(ptr nofreeobj noundef align 4 dereferenceable(4)) #[[ARGMEM_DECL]]
// LLVM: declare i32 @const_ignore() #[[NONE_DECL]]
// LLVM: declare i32 @const_coerced(i64) #[[NONE_DECL]]
// LLVM: declare i32 @const_flattened(i64, i64) #[[NONE_DECL]]

// The trailing wildcard covers target-features and the other codegen-option
// strings, which differ between the emits.
// LLVM-DAG: attributes #[[READ_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(read) {{.*}}}
// LLVM-DAG: attributes #[[NONE_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(none) {{.*}}}
// LLVM-DAG: attributes #[[NOALIAS_DEF]] = { {{.*}}nounwind{{.*}}memory(argmem: readwrite, inaccessiblemem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[ARGMEM_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[READ_ARGMEM_DEF]] = { {{.*}}nounwind{{.*}}willreturn memory(read, argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[ARGMEM_DECL]] = { nounwind willreturn memory(argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[READ_ARGMEM_DECL]] = { nounwind willreturn memory(read, argmem: readwrite) {{.*}}}
// LLVM-DAG: attributes #[[NONE_DECL]] = { nounwind willreturn memory(none) {{.*}}}
// LLVM-DAG: attributes #[[READ_CALL]] = { nounwind willreturn memory(read) }
// LLVM-DAG: attributes #[[NONE_CALL]] = { nounwind willreturn memory(none) }
// LLVM-DAG: attributes #[[NOALIAS_CALL]] = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }
// LLVM-DAG: attributes #[[ARGMEM_CALL]] = { nounwind willreturn memory(argmem: readwrite) }
// LLVM-DAG: attributes #[[READ_ARGMEM_CALL]] = { nounwind willreturn memory(read, argmem: readwrite) }
