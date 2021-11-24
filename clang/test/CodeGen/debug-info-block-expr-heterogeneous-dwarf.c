// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -gheterogeneous-dwarf -emit-llvm -disable-llvm-verifier -o - %s | FileCheck %s
// RUN: %clang_cc1 -DDEAD_CODE -fblocks -debug-info-kind=limited -gheterogeneous-dwarf -emit-llvm -disable-llvm-verifier -o - %s | FileCheck %s

typedef void (^BlockTy)();
void escapeFunc(BlockTy);
typedef void (^BlockTy)();
void noEscapeFunc(__attribute__((noescape)) BlockTy);

// Verify that the desired DIExpr are generated for escaping (i.e, not
// 'noescape') blocks.
void test_escape_func() {
// CHECK-LABEL: void @test_escape_func
// CHECK: call void @llvm.dbg.def(metadata ![[ESCAPE_VAR_LT:[0-9]+]], metadata %struct.__block_byref_escape_var* %escape_var), !dbg !{{[0-9]+}}
  __block int escape_var;
// Blocks in dead code branches still capture __block variables.
#ifdef DEAD_CODE
  if (0)
#endif
  escapeFunc(^{ (void)escape_var; });
}

// Verify that the desired DIExpr are generated for noescape blocks.
void test_noescape_func() {
// CHECK-LABEL: void @test_noescape_func
// CHECK: call void @llvm.dbg.def(metadata ![[NOESCAPE_VAR_LT:[0-9]+]], metadata i32* %noescape_var), !dbg !{{[0-9]+}}
  __block int noescape_var;
  noEscapeFunc(^{ (void)noescape_var; });
}

// Verify that the desired DIExpr are generated for blocks.
void test_local_block() {
// CHECK-LABEL: void @test_local_block
// CHECK: call void @llvm.dbg.def(metadata ![[BLOCK_VAR_LT:[0-9]+]], metadata %struct.__block_byref_block_var* %block_var), !dbg !{{[0-9]+}}
  __block int block_var;

// FIXME(KZHURAVL): Update EmitDeclareOfBlockDeclRefVariable and EmitDeclareOfBlockLiteralArgVariable.
// CHECK-LABEL: @__test_local_block_block_invoke
// CHECK: call void @llvm.dbg.declare({{.*}}!DIExpression(DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}){{.*}})
  ^ { block_var = 1; }();
}

// Verify that the desired DIExpr are generated for __block vars not used
// in any block.
void test_unused() {
// CHECK-LABEL: void @test_unused
// CHECK: call void @llvm.dbg.def(metadata ![[UNUSED_VAR_LT:[0-9]+]], metadata i32* %unused_var), !dbg !{{[0-9]+}}
  __block int unused_var;
// Use i (not inside a block).
  ++unused_var;
}

// CHECK-LABEL: !llvm.dbg.cu =

// CHECK-DAG: ![[ESCAPE_VAR_LT]] = distinct !DILifetime(object: ![[ESCAPE_VAR:[0-9]+]], location: !DIExpr(DIOpReferrer(%struct.__block_byref_escape_var*), DIOpDeref(i8*), DIOpConstant(i64 {{[0-9]+}}), DIOpByteOffset(i8*), DIOpDeref(i8*), DIOpConstant(i64 {{[0-9]+}}), DIOpByteOffset(i32)))
// CHECK-DAG: ![[ESCAPE_VAR]] = !DILocalVariable(name: "escape_var"
// CHECK-DAG: ![[NOESCAPE_VAR_LT]] = distinct !DILifetime(object: ![[NOESCAPE_VAR:[0-9]+]], location: !DIExpr(DIOpReferrer(i32*), DIOpDeref(i32)))
// CHECK-DAG: ![[NOESCAPE_VAR]] = !DILocalVariable(name: "noescape_var"
// CHECK-DAG: ![[BLOCK_VAR_LT]] = distinct !DILifetime(object: ![[BLOCK_VAR:[0-9]+]], location: !DIExpr(DIOpReferrer(%struct.__block_byref_block_var*), DIOpDeref(i8*), DIOpConstant(i64 {{[0-9]+}}), DIOpByteOffset(i8*), DIOpDeref(i8*), DIOpConstant(i64 {{[0-9]+}}), DIOpByteOffset(i32)))
// CHECK-DAG: ![[BLOCK_VAR]] = !DILocalVariable(name: "block_var"
// CHECK-DAG: ![[UNUSED_VAR_LT]] = distinct !DILifetime(object: ![[UNUSED_VAR:[0-9]+]], location: !DIExpr(DIOpReferrer(i32*), DIOpDeref(i32)))
// CHECK-DAG: ![[UNUSED_VAR]] = !DILocalVariable(name: "unused_var"
