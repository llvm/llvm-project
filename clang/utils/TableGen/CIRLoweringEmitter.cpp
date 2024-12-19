//===- CIRBuiltinsEmitter.cpp - Generate lowering of builtins --=-*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {
std::string ClassDeclaration;
std::string ClassDefinitions;
std::string ClassList;

std::string TBAANameFunctionDeclaration;
std::string TBAANameFunctionDefinitions;
std::string TBAANameClassList;

void GenerateLowering(const Record *Operation) {
  using namespace std::string_literals;
  std::string Name = Operation->getName().str();
  std::string LLVMOp = Operation->getValueAsString("llvmOp").str();

  ClassDeclaration +=
      "class CIR" + Name +
      "Lowering : public mlir::OpConversionPattern<cir::" + Name +
      R"C++(> {
  public:
    using OpConversionPattern<cir::)C++" +
      Name + R"C++(>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::)C++" +
      Name +
      " op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) "
      "const "
      "override;" +
      R"C++(
};
)C++";

  ClassDefinitions +=
      R"C++(mlir::LogicalResult
CIR)C++" +
      Name + "Lowering::matchAndRewrite(cir::" + Name +
      R"C++( op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const {)C++";

  auto ResultCount = Operation->getValueAsDag("results")->getNumArgs();
  if (ResultCount > 0)
    ClassDefinitions += R"C++(
  auto resTy = this->getTypeConverter()->convertType(op.getType());)C++";

  ClassDefinitions += R"C++(
  rewriter.replaceOpWithNewOp<mlir::LLVM::)C++" +
                      LLVMOp + ">(op";

  if (ResultCount > 0)
    ClassDefinitions += ", resTy";

  auto ArgCount = Operation->getValueAsDag("arguments")->getNumArgs();
  for (size_t i = 0; i != ArgCount; ++i)
    ClassDefinitions += ", adaptor.getOperands()[" + std::to_string(i) + ']';

  ClassDefinitions += R"C++();
  return mlir::success();
}
)C++";

  ClassList += ", CIR" + Name + "Lowering\n";
}

void GenerateTBAANameLowering(const Record *def) {
  using namespace std::string_literals;
  std::string Name = def->getValueAsString("cppClassName").str();
  std::string TBAAName = def->getValueAsString("tbaaName").str();
  TBAANameFunctionDeclaration += "llvm::StringRef getTBAAName(cir::";
  TBAANameFunctionDeclaration += Name + " ty);";
  TBAANameFunctionDeclaration += "\n";
  TBAANameFunctionDefinitions += "llvm::StringRef getTBAAName(cir::";
  TBAANameFunctionDefinitions += Name + " ty) {";
  TBAANameFunctionDefinitions += "  return \"" + TBAAName + "\";";
  TBAANameFunctionDefinitions += "}";
  TBAANameFunctionDefinitions += "\n";
  TBAANameClassList += "\n";
  TBAANameClassList += "cir::";
  TBAANameClassList += Name;
  TBAANameClassList += ", ";
}
} // namespace

void clang::EmitCIRBuiltinsLowering(const RecordKeeper &Records,
                                    raw_ostream &OS) {
  emitSourceFileHeader("Lowering of ClangIR builtins to LLVM IR builtins", OS);
  for (const auto *Builtin :
       Records.getAllDerivedDefinitions("LLVMLoweringInfo")) {
    if (!Builtin->getValueAsString("llvmOp").empty())
      GenerateLowering(Builtin);
  }

  OS << "#ifdef GET_BUILTIN_LOWERING_CLASSES_DECLARE\n"
     << ClassDeclaration << "\n#endif\n";
  OS << "#ifdef GET_BUILTIN_LOWERING_CLASSES_DEF\n"
     << ClassDefinitions << "\n#endif\n";
  OS << "#ifdef GET_BUILTIN_LOWERING_LIST\n" << ClassList << "\n#endif\n";
}

void clang::EmitCIRTBAANameLowering(const RecordKeeper &Records,
                                    raw_ostream &OS) {
  emitSourceFileHeader("Lowering of ClangIR TBAA Name", OS);

  for (const auto *Builtin :
       Records.getAllDerivedDefinitions("TBAALoweringInfo")) {
    if (!Builtin->getValueAsString("tbaaName").empty())
      GenerateTBAANameLowering(Builtin);
  }

  OS << "#ifdef GET_TBAANAME_LOWERING_FUNCTIONS_DECLARE\n"
     << TBAANameFunctionDeclaration << "\n#endif\n";
  OS << "#ifdef GET_TBAANAME_LOWERING_FUNCTIONS_DEF\n"
     << TBAANameFunctionDefinitions << "\n#endif\n";
  // remove last `, `
  if (!TBAANameClassList.empty()) {
    TBAANameClassList.resize(TBAANameClassList.size() - 2);
  }
  OS << "#ifdef GET_TBAANAME_LOWERING_LIST\n"
     << TBAANameClassList << "\n#endif\n";
}
