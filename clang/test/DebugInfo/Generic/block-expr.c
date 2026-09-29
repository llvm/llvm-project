// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DDEAD_CODE -fblocks -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

typedef void (^BlockTy)(void);
void escapeFunc(BlockTy);
typedef void (^BlockTy)(void);
void noEscapeFunc(__attribute__((noescape)) BlockTy);

// Verify that the desired DIExpression are generated for escaping (i.e, not
// 'noescape') blocks.
void test_escape_func(void) {
// CHECK-LABEL: void @test_escape_func
// CHECK: #dbg_declare({{.*}}![[ESCAPE_VAR:[0-9]+]], !DIExpression(DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}){{.*}})
  __block int escape_var;
// Blocks in dead code branches still capture __block variables.
#ifdef DEAD_CODE
  if (0)
#endif
  escapeFunc(^{ (void)escape_var; });
}

// Verify that the desired DIExpression are generated for noescape blocks.
void test_noescape_func(void) {
// CHECK-LABEL: void @test_noescape_func
// CHECK: #dbg_declare({{.*}}![[NOESCAPE_VAR:[0-9]+]], !DIExpression(),
  __block int noescape_var;
  noEscapeFunc(^{ (void)noescape_var; });
}

// Verify that the desired DIExpression are generated for blocks.
void test_local_block(void) {
// CHECK-LABEL: void @test_local_block
// CHECK: #dbg_declare({{.*}}![[BLOCK_VAR:[0-9]+]], !DIExpression(DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}){{.*}})
  __block int block_var;

// CHECK-LABEL: @__test_local_block_block_invoke
// CHECK: #dbg_declare({{.*}}!DIExpression(DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}){{.*}})
  ^ { block_var = 1; }();
}

// Verify that the desired DIExpression are generated for __block vars not used
// in any block.
void test_unused(void) {
// CHECK-LABEL: void @test_unused
// CHECK: #dbg_declare({{.*}}![[UNUSED_VAR:[0-9]+]], !DIExpression(),
  __block int unused_var;
// Use i (not inside a block).
  ++unused_var;
}

// CHECK: ![[ESCAPE_VAR]] = !DILocalVariable(name: "escape_var"
// CHECK: ![[NOESCAPE_VAR]] = !DILocalVariable(name: "noescape_var"
// CHECK: ![[BLOCK_VAR]] = !DILocalVariable(name: "block_var"
// CHECK: ![[UNUSED_VAR]] = !DILocalVariable(name: "unused_var"

