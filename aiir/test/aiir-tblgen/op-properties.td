// RUN: aiir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: aiir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEFS

include "aiir/IR/AttrTypeBase.td"
include "aiir/IR/EnumAttr.td"
include "aiir/IR/OpBase.td"
include "aiir/IR/Properties.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "foobar";
}
class NS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

def OpWithAttr : NS_Op<"op_with_attr">{
  let arguments = (ins AnyAttr:$attr, OptionalAttr<AnyAttr>:$optional);
}

// Test required and optional properties
// ---

def DefaultI64Array : IntArrayProp<I64Prop> {
  let defaultValue = "::llvm::ArrayRef<int64_t>{}";
  let storageTypeValueOverride = "::llvm::SmallVector<int64_t>{}";
}

def OpWithProps : NS_Op<"op_with_props"> {
  let arguments = (ins
    BoolProp:$flag,
    StringProp:$string,
    ArrayProp<StringProp>:$strings,
    DefaultValuedProp<I32Prop, "0">:$default_int,
    OptionalProp<I64Prop>:$optional,
    DefaultI64Array:$value
  );
}

/// Check that optional arguments to builders only go at the end.
def OpWithSomeOptionalProperties : NS_Op<"op_with_some_optional_props"> {
  let arguments = (ins
    OptionalProp<I64Prop>:$mustSpecify,
    I64Prop:$required,
    OptionalProp<StringProp>:$canOmit,
    DefaultValuedProp<I64Prop, "-1">:$canOmit2
  );
}

/// Check that the ambiguous attribute protection correctly stops optional properties
/// from getting default argument values in builders.
def OpWithOptionalPropsAndAttrs :
    NS_Op<"with_some_optional_props_and_atts"> {
  let arguments = (ins
    OptionalProp<BoolProp>:$mustSpecify,
    OptionalAttr<BoolAttr>:$ambiguous,
    OptionalAttr<I32Attr>:$canOmit,
    OptionalProp<I32Prop>:$canOmitProp
  );
}

// DECL: void setAttrAttr(::aiir::Attribute attr)
// DECL-NEXT: getProperties().attr = attr
// DECL: void setOptionalAttr(::aiir::Attribute attr)
// DECL-NEXT: getProperties().optional = attr

// -----

// DECL-LABEL: class OpWithOptionalPropsAndAttrs :
// DECL: static void build(
// DECL-SAME: ::aiir::OpBuilder &odsBuilder,
// DECL-SAME: ::aiir::OperationState &odsState,
// DECL-SAME: /*optional*/std::optional<bool> mustSpecify,
// DECL-SAME: /*optional*/::aiir::BoolAttr ambiguous,
// DECL-SAME: /*optional*/::aiir::IntegerAttr canOmit,
// DECL-SAME: /*optional*/std::optional<int32_t> canOmitProp = std::nullopt);

// -----

// COM: Ensure the struct is set up how we expect
// DECL-LABEL: class OpWithPropsGenericAdaptorBase
// DECL: using flagTy = bool;
// DECL-NEXT: flagTy flag;
// DECL-NEXT: bool getFlag()
// DECL-NEXT: propStorage = this->flag
// DECL-NEXT: return propStorage;
// DECL: void setFlag(bool propValue)
// DECL-NEXT: propStorage = this->flag;
// DECL-NEXT: propStorage = propValue;
// DECL: using stringTy = std::string;
// DECL: llvm::StringRef getString()
// DECL: auto &propStorage = this->string;
// DECL-NEXT: return ::llvm::StringRef{propStorage};
// DECL: using stringsTy = ::llvm::SmallVector<std::string>
// DECL: ::llvm::ArrayRef<std::string> getStrings()
// DECL: using default_intTy = int32_t;
// DECL: default_intTy default_int = 0;
// DECL: valueTy value = ::llvm::SmallVector<int64_t>{};
// DECL: ::llvm::ArrayRef<int64_t> getValue()
// DECL: return ::llvm::ArrayRef<int64_t>{propStorage}
// DECL: void setValue(::llvm::ArrayRef<int64_t> propValue)
// DECL: propStorage.assign
// DECL-LABEL: class OpWithProps :
// DECL: setString(::llvm::StringRef newString)
// DECL-NEXT: getProperties().setString(newString)

// DECL: static void build(
// DECL-SAME: ::aiir::OpBuilder &odsBuilder,
// DECL-SAME: ::aiir::OperationState &odsState,
// DECL-SAME: bool flag,
// DECL-SAME: ::llvm::StringRef string,
// DECL-SAME: ::llvm::ArrayRef<std::string> strings,
// DECL-SAME: /*optional*/int32_t default_int = 0,
// DECL-SAME: /*optional*/std::optional<int64_t> optional = std::nullopt,
// DECL-SAME: /*optional*/::llvm::ArrayRef<int64_t> value = ::llvm::ArrayRef<int64_t>{});

// DEFS-LABEL: OpWithProps::computePropertiesHash
// DEFS: hash_value_
// DEFS: using ::llvm::hash_value;
// DEFS-NEXT: return hash_value(::llvm::ArrayRef<int64_t>{propStorage})
// DEFS: hash_value(prop.optional)
// DEFS: hash_value_(prop.value)

// -----

// DECL-LABEL: class OpWithSomeOptionalProperties :
// DECL: static void build(
// DECL-SAME: ::aiir::OpBuilder &odsBuilder,
// DECL-SAME: ::aiir::OperationState &odsState,
// DECL-SAME: /*optional*/std::optional<int64_t> mustSpecify,
// DECL-SAME: int64_t required,
// DECL-SAME: /*optional*/std::optional<::llvm::StringRef> canOmit = std::nullopt,
// DECL-SAME: /*optional*/int64_t canOmit2 = -1);
