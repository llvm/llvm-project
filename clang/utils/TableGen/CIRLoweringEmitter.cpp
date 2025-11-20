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
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace clang;

namespace {
std::vector<std::string> LLVMLoweringPatterns;
std::vector<std::string> LLVMLoweringPatternsList;

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

std::string GetOpLLVMLoweringPatternName(llvm::StringRef OpName) {
  std::string Name = "CIRToLLVM";
  Name += OpName;
  Name += "Lowering";
  return Name;
}

void GenerateLLVMLoweringPattern(llvm::StringRef OpName,
                                 llvm::StringRef PatternName, bool IsRecursive,
                                 llvm::StringRef ExtraDecl) {
  std::string CodeBuffer;
  llvm::raw_string_ostream Code(CodeBuffer);

  Code << "class " << PatternName
       << " : public mlir::OpConversionPattern<cir::" << OpName << "> {\n";
  Code << "  [[maybe_unused]] mlir::DataLayout const &dataLayout;\n";
  Code << "\n";

  Code << "public:\n";
  Code << "  using mlir::OpConversionPattern<cir::" << OpName
       << ">::OpConversionPattern;\n";

  Code << "  " << PatternName
       << "(mlir::TypeConverter const "
          "&typeConverter, mlir::MLIRContext *context, mlir::DataLayout const "
          "&dataLayout)\n";
  Code << "    : OpConversionPattern<cir::" << OpName
       << ">(typeConverter, context), dataLayout(dataLayout)";
  if (IsRecursive) {
    Code << " {\n";
    Code << "    setHasBoundedRewriteRecursion();\n";
    Code << "  }\n";
  } else {
    Code << " {}\n";
  }

  Code << "\n";

  Code << "  mlir::LogicalResult matchAndRewrite(cir::" << OpName
       << " op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) "
          "const override;\n";

  if (!ExtraDecl.empty()) {
    Code << "\nprivate:\n";
    Code << ExtraDecl << "\n";
  }

  Code << "};\n";

  LLVMLoweringPatterns.push_back(std::move(CodeBuffer));
}

void Generate(const Record *OpRecord) {
  std::string OpName = GetOpCppClassName(OpRecord);

  if (OpRecord->getValueAsBit("hasLLVMLowering")) {
    std::string PatternName = GetOpLLVMLoweringPatternName(OpName);
    bool IsRecursive = OpRecord->getValueAsBit("isLLVMLoweringRecursive");
    llvm::StringRef ExtraDecl =
        OpRecord->getValueAsString("extraLLVMLoweringPatternDecl");

    GenerateLLVMLoweringPattern(OpName, PatternName, IsRecursive, ExtraDecl);
    LLVMLoweringPatternsList.push_back(std::move(PatternName));
  }
}
} // namespace

void clang::EmitCIRLowering(const llvm::RecordKeeper &RK,
                            llvm::raw_ostream &OS) {
  emitSourceFileHeader("Lowering patterns for CIR operations", OS);
  for (const auto *OpRecord : RK.getAllDerivedDefinitions("CIR_Op"))
    Generate(OpRecord);

  OS << "#ifdef GET_LLVM_LOWERING_PATTERNS\n"
     << llvm::join(LLVMLoweringPatterns, "\n") << "#endif\n\n";
  OS << "#ifdef GET_LLVM_LOWERING_PATTERNS_LIST\n"
     << llvm::join(LLVMLoweringPatternsList, ",\n") << "\n#endif\n\n";
}
