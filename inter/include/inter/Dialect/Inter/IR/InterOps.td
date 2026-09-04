// XW frontend operations.

#ifndef INTER_DIALECT_INTER_IR_INTEROPS_TD
#define INTER_DIALECT_INTER_IR_INTEROPS_TD

include "inter/Dialect/Inter/IR/InterBase.td"
include "inter/Dialect/Inter/IR/XWAttrs.td"
include "inter/Dialect/Inter/IR/XWEnums.td"
include "inter/Dialect/Inter/IR/XWTypes.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class XW_Op<string mnemonic, list<Trait> traits = []>
    : Op<XWDialect, mnemonic, traits> {
  let cppNamespace = "::xw";
}

def XW_CacheStateResource : Resource<"::xw::CacheStateResource">;

def XW_ConstantOp : XW_Op<"constant", [ConstantLike, Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static bool isBuildableWith(::mlir::Attribute value, ::mlir::Type type);
    static ConstantOp materialize(::mlir::OpBuilder &builder,
                                  ::mlir::Attribute value, ::mlir::Type type,
                                  ::mlir::Location loc);
  }];
}

def XW_SplatOp : XW_Op<"splat", [Pure]> {
  let arguments = (ins AnyType:$source);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_ReadFirstOp : XW_Op<"read_first", [NoMemoryEffect]> {
  let arguments = (ins XW_Simd:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_ExpandOp : XW_Op<"expand", [Pure]> {
  let arguments = (ins AnyType:$source);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_FreezeOp : XW_Op<"freeze", [AllTypesMatch<["source", "result"]>]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source)";
  let hasVerifier = 1;
}

def XW_BinaryOp : XW_Op<"binary", [Pure, Elementwise,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let arguments = (ins XW_BinaryKindEnum:$kind, AnyType:$lhs, AnyType:$rhs,
      DefaultValuedAttr<Arith_IntegerOverflowAttr,
          "::mlir::arith::IntegerOverflowFlags::none">:$overflowFlags);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind $lhs `,` $rhs (`overflow` `` $overflowFlags^)? attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_CastOp : XW_Op<"cast", [Pure, Elementwise,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let arguments = (ins XW_CastKindEnum:$kind, AnyType:$source,
      OptionalAttr<DictionaryAttr>:$policy,
      DefaultValuedAttr<Arith_IntegerOverflowAttr,
          "::mlir::arith::IntegerOverflowFlags::none">:$overflowFlags);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind $source (`policy` $policy^)? (`overflow` `` $overflowFlags^)? attr-dict `:` type($source) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_BitcastOp : XW_Op<"bitcast", [Pure]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}

class XW_FloatBinary<string mnemonic> : XW_Op<mnemonic, [Pure, Elementwise]> {
  let arguments = (ins XW_Simd:$lhs, XW_Simd:$rhs,
                       DefaultValuedAttr<Arith_FastMathAttr,
                           "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}
class XW_FloatUnary<string mnemonic> : XW_Op<mnemonic, [Pure, Elementwise]> {
  let arguments = (ins XW_Simd:$source,
                       DefaultValuedAttr<Arith_FastMathAttr,
                           "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$source (`fastmath` `` $fastmath^)? attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}
def XW_FAddOp : XW_FloatBinary<"fadd">;
def XW_FSubOp : XW_FloatBinary<"fsub">;
def XW_FMulOp : XW_FloatBinary<"fmul">;
def XW_FMaxOp : XW_FloatBinary<"fmax">;
def XW_FmaOp : XW_Op<"fma", [Pure, Elementwise]> {
  let arguments = (ins XW_Simd:$lhs, XW_Simd:$rhs, XW_Simd:$acc,
                       DefaultValuedAttr<Arith_FastMathAttr,
                           "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $acc (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)";
  let hasVerifier = 1;
}
def XW_FExp2Op : XW_FloatUnary<"fexp2">;
def XW_FRcpOp : XW_FloatUnary<"frcp">;

def XW_DpasOp : XW_Op<"dpas", [Pure]> {
  let summary = "Xe DPAS matrix multiply-accumulate";
  let arguments = (ins XW_Simd:$a, XW_Simd:$b, XW_Simd:$acc,
                       XW_DpasPrecisionEnum:$a_precision,
                       XW_DpasPrecisionEnum:$b_precision,
                       I64Attr:$k,
                       I64Attr:$systolic_depth, I64Attr:$repeat_count);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc) `->` type($result)";
  let hasVerifier = 1;
}

def XW_CmpIOp : XW_Op<"cmpi", [Pure, Elementwise,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate, AnyType:$lhs,
                       AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}
def XW_CmpFOp : XW_Op<"cmpf", [Pure, Elementwise]> {
  let arguments = (ins Arith_CmpFPredicateAttr:$predicate, AnyType:$lhs,
                       AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}
def XW_SelectOp : XW_Op<"select", [Pure, Elementwise,
    AllTypesMatch<["true_value", "false_value", "result"]>]> {
  let arguments = (ins AnyType:$condition, AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($condition) `,` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

class XW_MaskBinary<string mnemonic> : XW_Op<mnemonic, [Pure, Elementwise,
    AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins XW_Mask:$lhs, XW_Mask:$rhs);
  let results = (outs XW_Mask:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}
def XW_MaskAndOp : XW_MaskBinary<"mask_and">;
def XW_MaskOrOp : XW_MaskBinary<"mask_or">;
def XW_MaskXorOp : XW_MaskBinary<"mask_xor">;
def XW_MaskNotOp : XW_Op<"mask_not", [Pure, Elementwise,
    AllTypesMatch<["input", "result"]>]> {
  let arguments = (ins XW_Mask:$input);
  let results = (outs XW_Mask:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}
def XW_BallotOp : XW_Op<"ballot", [NoMemoryEffect]> {
  let arguments = (ins XW_Mask:$mask);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$mask attr-dict `:` qualified(type($mask)) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_PackOp : XW_Op<"pack", [Pure, SameTypeOperands]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs XW_VectorOrSimdVector:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}
def XW_ExtractOp : XW_Op<"extract", [Pure]> {
  let arguments = (ins XW_VectorOrSimdVector:$source,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source `[` $index `]` attr-dict `:` type($source) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_PtrAddOp : XW_Op<"ptradd", [Pure, Elementwise,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let arguments = (ins AnyType:$base, AnyType:$offset);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `,` $offset attr-dict `:` type($base) `,` type($offset) `->` type($result)";
  let hasVerifier = 1;
}
def XW_AddrspaceCastOp : XW_Op<"addrspace_cast", [Pure, Elementwise,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}
def XW_PtrToIntOp : XW_Op<"ptr_to_int", [Pure, Elementwise]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}
def XW_IntToPtrOp : XW_Op<"int_to_ptr", [Pure, Elementwise]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}
def XW_NullOp : XW_Op<"null", [ConstantLike, Pure]> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}
def XW_PtrCmpOp : XW_Op<"ptr_cmp", [Pure, Elementwise]> {
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate, AnyType:$lhs,
                       AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def XW_YieldOp : XW_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["WhereOp"]>,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                              ["getMutableSuccessorOperands"]>]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange{}); }]>];
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}
def XW_WhereOp : XW_Op<"where", [
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorInputs"]>,
    RecursiveMemoryEffects, NoRegionArguments]> {
  let arguments = (ins XW_Mask:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, MaxSizedRegion<1>:$elseRegion);
  let assemblyFormat = "$condition $thenRegion (`otherwise` $elseRegion^)? attr-dict `:` qualified(type($condition)) (`->` type($results)^)?";
  let hasVerifier = 1;
}

class XW_DimQueryOp<string mnemonic, string summaryText>
    : XW_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins I64Attr:$dim);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
  let hasVerifier = 1;
}
def XW_LaneIdOp : XW_Op<"lane_id", [Pure]> {
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}
def XW_SubgroupIdOp : XW_Op<"subgroup_id", [Pure]> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}
def XW_GlobalIdOp : XW_DimQueryOp<"global_id", "global invocation id">;
def XW_LocalIdOp : XW_DimQueryOp<"local_id", "local invocation id">;
def XW_GroupIdOp : XW_DimQueryOp<"group_id", "workgroup id">;
def XW_GlobalSizeOp : XW_DimQueryOp<"global_size", "global launch size">;
def XW_LocalSizeOp : XW_DimQueryOp<"local_size", "local launch size">;
def XW_NumGroupsOp : XW_DimQueryOp<"num_groups", "launch workgroup count">;
def XW_LaunchGridSizeOp
    : XW_DimQueryOp<"launch_grid_size", "launch grid size">;
def XW_LaunchBlockSizeOp
    : XW_DimQueryOp<"launch_block_size", "launch block size">;

def XW_ShuffleOp : XW_Op<"shuffle", [NoMemoryEffect]> {
  let arguments = (ins XW_Simd:$source, AnyType:$source_lane);
  let results = (outs XW_Simd:$result);
  let assemblyFormat = "$source `from` $source_lane attr-dict `:` type($source) `,` type($source_lane) `->` type($result)";
  let hasVerifier = 1;
}

def XW_TokenOp : XW_Op<"token", [Pure]> {
  let results = (outs XW_MemToken:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}
class XW_TokenMergeOp<string mnemonic> : XW_Op<mnemonic, [Pure]> {
  let arguments = (ins Variadic<XW_MemToken>:$dependencies);
  let results = (outs XW_MemToken:$result);
  let assemblyFormat = "$dependencies attr-dict `:` type($dependencies) `->` qualified(type($result))";
  let hasFolder = 1;
}
def XW_IssueTokenOp : XW_TokenMergeOp<"issue_token">;
def XW_AfterOp : XW_TokenMergeOp<"after">;
def XW_JoinOp : XW_TokenMergeOp<"join">;

def XW_LoadOp : XW_Op<"load"> {
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$ptr,
                       Optional<XW_MemToken>:$dependency);
  let results = (outs AnyType:$value, XW_MemToken:$token);
  let assemblyFormat = "$ptr (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def XW_StoreOp : XW_Op<"store"> {
  let arguments = (ins AnyType:$value, Arg<AnyType, "", [MemWrite]>:$ptr,
                       Optional<XW_MemToken>:$dependency);
  let results = (outs XW_MemToken:$token);
  let assemblyFormat = "$value `->` $ptr (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
class XW_Block2DOp<string mnemonic, list<Trait> traits = []>
    : XW_Op<mnemonic, traits> {
  let arguments = (ins Arg<XW_Ptr, "", [MemRead]>:$base,
                       AnySignlessInteger:$surface_width,
                       AnySignlessInteger:$surface_height,
                       AnySignlessInteger:$surface_pitch,
                       AnySignlessInteger:$x, AnySignlessInteger:$y,
                       I64Attr:$element_bits,
                       I64Attr:$block_width, I64Attr:$block_height,
                       I64Attr:$blocks, UnitAttr:$transpose, UnitAttr:$vnni,
                       Optional<XW_MemToken>:$dependency);
}
def XW_Block2DPrefetchOp : XW_Block2DOp<"block2d_prefetch",
    [MemoryEffects<[MemWrite<XW_CacheStateResource>]>]> {
  let results = (outs XW_MemToken:$token);
  let assemblyFormat = "$base `[` $x `,` $y `]` `surface` `(` $surface_width `,` $surface_height `,` $surface_pitch `)` (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def XW_Block2DReadOp : XW_Block2DOp<"block2d_read"> {
  let results = (outs XW_Simd:$value, XW_MemToken:$token);
  let assemblyFormat = "$base `[` $x `,` $y `]` `surface` `(` $surface_width `,` $surface_height `,` $surface_pitch `)` (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def XW_Block2DWriteOp : XW_Op<"block2d_write"> {
  let arguments = (ins XW_Simd:$value,
                       Arg<XW_Ptr, "", [MemWrite]>:$base,
                       AnySignlessInteger:$surface_width,
                       AnySignlessInteger:$surface_height,
                       AnySignlessInteger:$surface_pitch,
                       AnySignlessInteger:$x, AnySignlessInteger:$y,
                       I64Attr:$element_bits,
                       I64Attr:$block_width, I64Attr:$block_height,
                       I64Attr:$blocks, UnitAttr:$transpose, UnitAttr:$vnni,
                       Optional<XW_MemToken>:$dependency);
  let results = (outs XW_MemToken:$token);
  let assemblyFormat = "$value `->` $base `[` $x `,` $y `]` `surface` `(` $surface_width `,` $surface_height `,` $surface_pitch `)` (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def XW_AtomicRMWOp : XW_Op<"atomic_rmw"> {
  let arguments = (ins AtomicRMWKindAttr:$kind, AnyType:$value,
                       Arg<AnyType, "", [MemRead, MemWrite]>:$ptr,
                       Optional<XW_MemToken>:$dependency);
  let results = (outs AnyType:$old, XW_MemToken:$token);
  let assemblyFormat = "$kind $value `,` $ptr (`after` $dependency^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def XW_BarrierOp : XW_Op<"barrier", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins Variadic<XW_MemToken>:$dependencies);
  let results = (outs XW_MemToken:$token);
  let assemblyFormat = "$dependencies attr-dict `:` type($dependencies) `->` qualified(type($token))";
}
def XW_LocalMemoryBaseOp : XW_Op<"local_memory_base", [Pure]> {
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$offset);
  let results = (outs XW_Ptr:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}
def XW_AllocOp : XW_Op<"alloc"> {
  let arguments = (ins I64Attr:$bytesize, I64Attr:$align,
                       OptionalAttr<I64Attr>:$offset);
  let results = (outs Res<XW_Ptr, "", [MemAlloc]>:$result);
  let assemblyFormat = "`(` `)` attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}
def XW_AllocReleaseOp : XW_Op<"alloc_release"> {
  let arguments = (ins Arg<XW_Ptr, "", [MemFree]>:$allocation,
                       XW_MemToken:$dependency);
  let results = (outs XW_MemToken:$token);
  let assemblyFormat = "$allocation `after` $dependency attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // INTER_DIALECT_INTER_IR_INTEROPS_TD
