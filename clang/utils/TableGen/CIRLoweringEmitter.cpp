//===-- CIRLoweringEmitter.cpp - Generate CIR lowering patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This TableGen backend emits CIR operation lowering patterns.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace clang;

namespace {
std::vector<std::string> CXXABILoweringPatterns;
std::vector<std::string> CXXABILoweringPatternsList;
std::vector<std::string> CXXABILoweringAttrAlwaysLegal;
std::vector<std::string> LLVMLoweringPatterns;
std::vector<std::string> LLVMLoweringPatternsList;
std::string CIRAttrToValueVisitFunc;
std::vector<std::string> CIRAttrToValueVisitorCaseTypes;
std::vector<std::string> CIRAttrToValueVisitorDecls;

struct CustomLoweringCtor {
  struct Param {
    std::string Type;
    std::string Name;
  };

  std::vector<Param> Params;
};

// Adapted from mlir/lib/TableGen/Operator.cpp
// Returns the C++ class name of the operation, which is the name of the
// operation with the dialect prefix removed and the first underscore removed.
// If the operation name starts with an underscore, the underscore is considered
// part of the class name.
std::string GetOpCppClassName(const Record *OpRecord) {
  StringRef Name = OpRecord->getName();
  StringRef Prefix;
  StringRef CppClassName;
  std::tie(Prefix, CppClassName) = Name.split('_');
  if (Prefix.empty()) {
    // Class name with a leading underscore and without dialect prefix
    return Name.str();
  }
  if (CppClassName.empty()) {
    // Class name without dialect prefix
    return Prefix.str();
  }

  return CppClassName.str();
}

std::string GetOpABILoweringPatternName(llvm::StringRef OpName) {
  std::string Name = "CIR";
  Name += OpName;
  Name += "ABILowering";
  return Name;
}

std::string GetOpLLVMLoweringPatternName(llvm::StringRef OpName) {
  std::string Name = "CIRToLLVM";
  Name += OpName;
  Name += "Lowering";
  return Name;
}
std::optional<CustomLoweringCtor> parseCustomLoweringCtor(const Record *R) {
  if (!R)
    return std::nullopt;

  CustomLoweringCtor Ctor;
  const DagInit *Args = R->getValueAsDag("dagParams");

  for (const auto &[Arg, Name] : Args->getArgAndNames()) {
    Ctor.Params.push_back(
        {Arg->getAsUnquotedString(), Name->getAsUnquotedString()});
  }

  return Ctor;
}

void emitCustomParamList(raw_ostream &Code,
                         ArrayRef<CustomLoweringCtor::Param> Params) {
  for (const CustomLoweringCtor::Param &Param : Params) {
    Code << ", ";
    Code << Param.Type << " " << Param.Name;
  }
}

void emitCustomInitList(raw_ostream &Code,
                        ArrayRef<CustomLoweringCtor::Param> Params) {
  for (const CustomLoweringCtor::Param &P : Params)
    Code << ", " << P.Name << "(" << P.Name << ")";
}

void GenerateABILoweringPattern(llvm::StringRef OpName,
                                llvm::StringRef PatternName) {
  std::string CodeBuffer;
  llvm::raw_string_ostream Code(CodeBuffer);

  Code << "class " << PatternName
       << " : public mlir::OpConversionPattern<cir::" << OpName << "> {\n";
  Code << "  [[maybe_unused]] mlir::DataLayout *dataLayout;\n";
  Code << "  [[maybe_unused]] cir::LowerModule *lowerModule;\n";
  Code << "\n";

  Code << "public:\n";
  Code << "  " << PatternName
       << "(mlir::MLIRContext *context, const mlir::TypeConverter "
          "&typeConverter, mlir::DataLayout &dataLayout, cir::LowerModule "
          "&lowerModule)\n";
  Code << "    : OpConversionPattern<cir::" << OpName
       << ">(typeConverter, context), dataLayout(&dataLayout), "
          "lowerModule(&lowerModule) {}\n";
  Code << "\n";

  Code << "  mlir::LogicalResult matchAndRewrite(cir::" << OpName
       << " op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) "
          "const override;\n";

  Code << "};\n";

  CXXABILoweringPatterns.push_back(std::move(CodeBuffer));
}

void GenerateLLVMLoweringPattern(llvm::StringRef OpName,
                                 llvm::StringRef PatternName, bool IsRecursive,
                                 llvm::StringRef ExtraDecl,
                                 const Record *CustomCtorRec,
                                 llvm::StringRef LLVMOp) {
  std::optional<CustomLoweringCtor> CustomCtor =
      parseCustomLoweringCtor(CustomCtorRec);
  std::string CodeBuffer;
  llvm::raw_string_ostream Code(CodeBuffer);

  Code << "class " << PatternName
       << " : public mlir::OpConversionPattern<cir::" << OpName << "> {\n";
  Code << "  [[maybe_unused]] mlir::DataLayout const &dataLayout;\n";

  if (CustomCtor) {
    for (const CustomLoweringCtor::Param &P : CustomCtor->Params)
      Code << "  " << P.Type << " " << P.Name << ";\n";
  }

  Code << "\n";

  Code << "public:\n";
  Code << "  using mlir::OpConversionPattern<cir::" << OpName
       << ">::OpConversionPattern;\n";

  // Constructor
  Code << "  " << PatternName
       << "(const mlir::TypeConverter &typeConverter, "
          "mlir::MLIRContext *context, const mlir::DataLayout &dataLayout";

  if (CustomCtor)
    emitCustomParamList(Code, CustomCtor->Params);

  Code << ")\n";

  Code << "    : OpConversionPattern<cir::" << OpName
       << ">(typeConverter, context), dataLayout(dataLayout)";

  if (CustomCtor)
    emitCustomInitList(Code, CustomCtor->Params);

  Code << " {\n";

  if (IsRecursive)
    Code << "    setHasBoundedRewriteRecursion();\n";

  Code << "  }\n\n";

  if (!LLVMOp.empty()) {
    // Generate the matchAndRewrite body automatically.
    Code
        << "  mlir::LogicalResult matchAndRewrite(cir::" << OpName
        << " op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) "
           "const override {\n";
    Code
        << "    mlir::Type resTy = typeConverter->convertType(op.getType());\n";
    Code << "    rewriter.replaceOpWithNewOp<mlir::LLVM::" << LLVMOp
         << ">(op, resTy, adaptor.getOperands());\n";
    Code << "    return mlir::success();\n";
    Code << "  }\n";
  } else {
    Code
        << "  mlir::LogicalResult matchAndRewrite(cir::" << OpName
        << " op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) "
           "const override;\n";
  }

  if (!ExtraDecl.empty()) {
    Code << "\nprivate:\n";
    Code << ExtraDecl << "\n";
  }

  Code << "};\n";

  LLVMLoweringPatterns.push_back(std::move(CodeBuffer));
}

void Generate(const Record *OpRecord) {
  std::string OpName = GetOpCppClassName(OpRecord);

  if (OpRecord->getValueAsBit("hasCXXABILowering")) {
    std::string PatternName = GetOpABILoweringPatternName(OpName);
    GenerateABILoweringPattern(OpName, PatternName);
    CXXABILoweringPatternsList.push_back(std::move(PatternName));
  }

  if (OpRecord->getValueAsBit("hasLLVMLowering")) {
    std::string PatternName = GetOpLLVMLoweringPatternName(OpName);
    bool IsRecursive = OpRecord->getValueAsBit("isLLVMLoweringRecursive");
    const Record *CustomCtor =
        OpRecord->getValueAsOptionalDef("customLLVMLoweringConstructorDecl");
    llvm::StringRef ExtraDecl =
        OpRecord->getValueAsString("extraLLVMLoweringPatternDecl");

    llvm::StringRef LLVMOp = OpRecord->getValueAsString("llvmOp");

    if (!LLVMOp.empty() && CustomCtor)
      PrintFatalError(OpRecord->getLoc(),
                      "op '" + OpName +
                          "' has both llvmOp and a custom lowering "
                          "constructor, which is not supported");

    GenerateLLVMLoweringPattern(OpName, PatternName, IsRecursive, ExtraDecl,
                                CustomCtor, LLVMOp);
    // Only automatically register patterns that use the default constructor.
    // Patterns with a custom constructor must be manually registered by the
    // lowering pass.
    if (!CustomCtor)
      LLVMLoweringPatternsList.push_back(std::move(PatternName));
  }
}

void GenerateCIREnumAttrs(const Record *Record) {
  std::string OpName = GetOpCppClassName(Record);
  // EnumAttr is in a separate hierarchy, so we have to set these separately, as
  // they never have an 'illegal' CXXABI type in them.
  CXXABILoweringAttrAlwaysLegal.push_back("cir::" + OpName);
}

void GenerateAttrToValueVisitor(const Record *Rec) {
  const Record *DialectRec = Rec->getValueAsDef("dialect");
  llvm::StringRef Ns = DialectRec->getValueAsString("cppNamespace");
  Ns.consume_front("::");
  std::string CppClassRef = Ns.str();
  CppClassRef += "::";
  CppClassRef += Rec->getValueAsString("cppClassName");

  std::string CodeBuffer;
  llvm::raw_string_ostream Code(CodeBuffer);
  Code << "  mlir::Value visitCirAttr(" << CppClassRef << " attr);";
  CIRAttrToValueVisitorDecls.push_back(std::move(CodeBuffer));
  CIRAttrToValueVisitorCaseTypes.push_back(std::move(CppClassRef));
}

void GenerateAttrToValueVisitFunc() {
  std::string CodeBuffer;
  llvm::raw_string_ostream Code(CodeBuffer);
  Code << "  mlir::Value visit(mlir::Attribute attr) {\n"
       << "    return llvm::TypeSwitch<mlir::Attribute, mlir::Value>(attr)\n"
       << "        .Case<\n              "
       << llvm::join(CIRAttrToValueVisitorCaseTypes, ",\n              ")
       << ">(\n"
       << "            [&](auto attrT) { return visitCirAttr(attrT); })\n"
       << "        .Default([this](mlir::Attribute attr) {\n"
       << "          mlir::emitError(parentOp->getLoc(), \"unsupported CIR "
          "attribute in LLVM constant lowering\")\n"
       << "              << attr;\n"
       << "          return mlir::Value();\n"
       << "        });\n"
       << "  }\n";
  CIRAttrToValueVisitFunc = std::move(CodeBuffer);
}

void GenerateCIRAttrs(const Record *Record) {
  std::string OpName = GetOpCppClassName(Record);
  if (!Record->getValueAsBit("canHaveIllegalCXXABIType"))
    CXXABILoweringAttrAlwaysLegal.push_back("cir::" + OpName);
  if (Record->getValueAsBit("hasAttrToValueLowering"))
    GenerateAttrToValueVisitor(Record);
}
} // namespace

void clang::EmitCIRLowering(const llvm::RecordKeeper &RK,
                            llvm::raw_ostream &OS) {
  emitSourceFileHeader("Lowering patterns for CIR operations", OS);
  for (const auto *OpRecord : RK.getAllDerivedDefinitions("CIR_Op"))
    Generate(OpRecord);
  for (const auto *OpRecord : RK.getAllDerivedDefinitions("EnumAttr"))
    GenerateCIREnumAttrs(OpRecord);
  for (const auto *OpRecord : RK.getAllDerivedDefinitions("CIR_Attr"))
    GenerateCIRAttrs(OpRecord);
  GenerateAttrToValueVisitFunc();

  OS << "#ifdef GET_CIR_ATTR_TO_VALUE_VISITOR_DECLS\n"
     << CIRAttrToValueVisitFunc << "\n"
     << llvm::join(CIRAttrToValueVisitorDecls, "\n") << "\n"
     << "#endif // GET_CIR_ATTR_TO_VALUE_VISITOR_DECLS\n\n";

  OS << "#ifdef GET_ABI_LOWERING_PATTERNS\n"
     << llvm::join(CXXABILoweringPatterns, "\n") << "#endif\n\n";
  OS << "#ifdef GET_ABI_LOWERING_PATTERNS_LIST\n"
     << llvm::join(CXXABILoweringPatternsList, ",\n") << "\n#endif\n\n";

  OS << "#ifdef GET_LLVM_LOWERING_PATTERNS\n"
     << llvm::join(LLVMLoweringPatterns, "\n") << "#endif\n\n";
  OS << "#ifdef GET_LLVM_LOWERING_PATTERNS_LIST\n"
     << llvm::join(LLVMLoweringPatternsList, ",\n") << "\n#endif\n\n";

  OS << "#ifdef CXX_ABI_ALWAYS_LEGAL_ATTRS\n"
     << llvm::join(CXXABILoweringAttrAlwaysLegal, ",\n") << "\n#endif\n\n";
}
