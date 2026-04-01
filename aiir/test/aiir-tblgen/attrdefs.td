// RUN: aiir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: aiir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "aiir/IR/AttrTypeBase.td"
include "aiir/IR/OpBase.td"

// DECL: #ifdef GET_ATTRDEF_CLASSES
// DECL: #undef GET_ATTRDEF_CLASSES

// DECL: namespace aiir {
// DECL: class AsmParser;
// DECL: class AsmPrinter;
// DECL: } // namespace aiir

// DEF: #ifdef GET_ATTRDEF_LIST
// DEF: #undef GET_ATTRDEF_LIST
// DEF: ::test::IndexAttr,
// DEF: ::test::SimpleAAttr,
// DEF: ::test::CompoundAAttr,
// DEF: ::test::SingleParameterAttr

// DEF-LABEL: ::aiir::OptionalParseResult generatedAttributeParser(
// DEF-SAME: ::aiir::AsmParser &parser,
// DEF-SAME: ::llvm::StringRef *mnemonic, ::aiir::Type type,
// DEF-SAME: ::aiir::Attribute &value) {
// DEF: return ::aiir::AsmParser::KeywordSwitch<::aiir::OptionalParseResult>(parser)
// DEF: .Case(::test::IndexAttr::getMnemonic()
// DEF-NEXT:   value = ::test::IndexAttr::parse(parser, type);
// DEF-NEXT:   return ::aiir::success(!!value);
// DEF: .Case(::test::CompoundAAttr::getMnemonic()
// DEF-NEXT: value = ::test::CompoundAAttr::parse(parser, type);
// DEF-NEXT: return ::aiir::success(!!value);
// DEF-NEXT: })
// DEF: .Default([&](llvm::StringRef keyword,
// DEF-NEXT:   *mnemonic = keyword;
// DEF-NEXT:   return std::nullopt;

def Test_Dialect: Dialect {
// DECL-NOT: TestDialect
// DEF-NOT: TestDialect
    let name = "TestDialect";
    let cppNamespace = "::test";
}

class TestAttr<string name> : AttrDef<Test_Dialect, name> { }

def C_IndexAttr : TestAttr<"Index"> {
    let mnemonic = "index";

    let parameters = (
      ins
      StringRefParameter<"Label for index">:$label
    );
  let hasCustomAssemblyFormat = 1;

// DECL-LABEL: class IndexAttr : public ::aiir::Attribute
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"index"};
// DECL: }
// DECL: static ::aiir::Attribute parse(
// DECL-SAME: ::aiir::AsmParser &odsParser, ::aiir::Type odsType);
// DECL: void print(::aiir::AsmPrinter &odsPrinter) const;
}

def A_SimpleAttrA : TestAttr<"SimpleA"> {
  let attrName = "test.simple";
// DECL: class SimpleAAttr : public ::aiir::Attribute
}

// A more complex parameterized type
def B_CompoundAttrA : TestAttr<"CompoundA"> {
  let summary = "A more complex parameterized attribute";
  let description = "This attribute is to test a reasonably complex attribute";
  let mnemonic = "cmpnd_a";
  let parameters = (
      ins
      "int":$widthOfSomething,
      "::test::SimpleTypeA": $exampleTdType,
      APFloatParameter<"">: $apFloat,
      ArrayRefParameter<"int", "Matrix dimensions">:$dims,
      AttributeSelfTypeParameter<"">:$inner
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

// DECL-LABEL: class CompoundAAttr : public ::aiir::Attribute
// DECL: static CompoundAAttr getChecked(::llvm::function_ref<::aiir::InFlightDiagnostic()> emitError, ::aiir::AIIRContext *context, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::aiir::Type inner);
// DECL: static ::llvm::LogicalResult verify(::llvm::function_ref<::aiir::InFlightDiagnostic()> emitError, int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::aiir::Type inner);
// DECL: static constexpr ::llvm::StringLiteral getMnemonic() {
// DECL:   return {"cmpnd_a"};
// DECL: }
// DECL: static ::aiir::Attribute parse(
// DECL-SAME: ::aiir::AsmParser &odsParser, ::aiir::Type odsType);
// DECL: void print(::aiir::AsmPrinter &odsPrinter) const;
// DECL: int getWidthOfSomething() const;
// DECL: ::test::SimpleTypeA getExampleTdType() const;
// DECL: ::llvm::APFloat getApFloat() const;
// DECL: ::aiir::Type getInner() const;

// Check that AttributeSelfTypeParameter is handled properly.
// DEF-LABEL: struct CompoundAAttrStorage
// DEF: CompoundAAttrStorage(
// DEF-SAME: inner(std::move(inner))

// DEF: bool operator==(const KeyTy &tblgenKey) const {
// DEF-NEXT: return
// DEF-SAME: (widthOfSomething == std::get<0>(tblgenKey)) &&
// DEF-SAME: (exampleTdType == std::get<1>(tblgenKey)) &&
// DEF-SAME: (apFloat.bitwiseIsEqual(std::get<2>(tblgenKey))) &&
// DEF-SAME: (dims == std::get<3>(tblgenKey)) &&
// DEF-SAME: (inner == std::get<4>(tblgenKey));

// DEF: static CompoundAAttrStorage *construct
// DEF: return new (allocator.allocate<CompoundAAttrStorage>())
// DEF-SAME: CompoundAAttrStorage(std::move(widthOfSomething), std::move(exampleTdType), std::move(apFloat), std::move(dims), std::move(inner));

// DEF: CompoundAAttr CompoundAAttr::getChecked(
// DEF-SAME:   int widthOfSomething, ::test::SimpleTypeA exampleTdType, ::llvm::APFloat apFloat, ::llvm::ArrayRef<int> dims, ::aiir::Type inner
// DEF-SAME: )
// DEF-NEXT: return Base::getChecked(emitError, context, std::move(widthOfSomething), std::move(exampleTdType), std::move(apFloat), std::move(dims), std::move(inner));

// DEF: ::aiir::Type CompoundAAttr::getInner() const {
// DEF-NEXT: return getImpl()->inner;
}

def D_SingleParameterAttr : TestAttr<"SingleParameter"> {
  let parameters = (
    ins
    "int": $num
  );
  let attrName = "test.single_parameter";
// DECL-LABEL: struct SingleParameterAttrStorage;
// DECL-LABEL: class SingleParameterAttr
// DECL-SAME:  detail::SingleParameterAttrStorage
}

def F_ParamWithAccessorTypeAttr : TestAttr<"ParamWithAccessorType"> {
  let parameters = (ins AttrParameter<"std::string", "", "StringRef">:$param);
  let attrName = "test.param_with_accessor_type";
}

// DECL-LABEL: class ParamWithAccessorTypeAttr
// DECL: StringRef getParam()
// DEF: ParamWithAccessorTypeAttrStorage
// DEF: ParamWithAccessorTypeAttrStorage(std::string param)
// DEF: StringRef ParamWithAccessorTypeAttr::getParam()

def G_BuilderWithReturnTypeAttr : TestAttr<"BuilderWithReturnType"> {
  let parameters = (ins "int":$a);
  let genVerifyDecl = 1;
  let builders = [AttrBuilder<(ins), [{ return {}; }], "::aiir::Attribute">];
  let attrName = "test.builder_with_return_type";
}

// DECL-LABEL: class BuilderWithReturnTypeAttr
// DECL: ::aiir::Attribute get(
// DECL: ::aiir::Attribute getChecked(

def H_TestExtraClassAttr : TestAttr<"TestExtraClass"> {
  let extraClassDeclaration = [{
    /// Test Method
    static int getFoo(int i);
  }];
  let extraClassDefinition = [{
    int $cppClass::getFoo(int i) {
      return i+1;
    }
  }];
  let attrName = "test.test_extra_class";
}

// DECL-LABEL: TestExtraClassAttr : public ::aiir::Attribute
// DECL: /// Test Method
// DECL-NEXT: static int getFoo(int i);

// DEF-LABEL: int TestExtraClassAttr::getFoo(int i) {
// DEF: return i+1;
// DEF-NEXT: }

def I_TestGenMnemonicAliasAttr : TestAttr<"TestGenMnemonicAlias"> {
  let mnemonic = "test_gen_mnemonic_alias";
  let genMnemonicAlias = 1;
}

// DECL-LABEL: class TestGenMnemonicAliasAttr : public ::aiir::Attribute
// DECL-SAME: ::aiir::OpAsmAttrInterface::Trait
// DECL: ::aiir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const;

// DEF-LABEL: ::aiir::OpAsmAliasResult TestGenMnemonicAliasAttr::getAlias(::llvm::raw_ostream &os) const {
// DEF-NEXT: os << "test_gen_mnemonic_alias";
// DEF-NEXT: return ::aiir::OpAsmAliasResult::OverridableAlias;
// DEF-NEXT: }

def J_CustomStorageCtorAttr : AttrDef<Test_Dialect, "CustomStorageCtorAttr"> {
  let attrName = "test_custom_storage_ctor_attr";
  let parameters = (ins "bool":$flag);
  let hasStorageCustomConstructor = 1;
}

// Note: ';' at the end of construct method declaration is important - otherwise
// one cannot provide custom definition

// DEF-LABEL: struct CustomStorageCtorAttrAttrStorage : public ::aiir::AttributeStorage
// DEF: static CustomStorageCtorAttrAttrStorage *construct
// DEF-SAME: (::aiir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey);
