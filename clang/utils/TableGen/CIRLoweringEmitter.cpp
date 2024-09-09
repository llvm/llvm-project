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
std::string ClassDefinitions;
std::string ClassList;

void GenerateLowering(const Record *Operation) {
  using namespace std::string_literals;
  std::string Name = Operation->getName().str();
  std::string LLVMOp = Operation->getValueAsString("llvmOp").str();
  ClassDefinitions +=
      "class CIR" + Name +
      "Lowering : public mlir::OpConversionPattern<mlir::cir::" + Name +
      R"C++(> {
  public:
    using OpConversionPattern<mlir::cir::)C++" +
      Name + R"C++(>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::)C++" +
      Name +
      " op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) "
      "const "
      "override {";

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
};
)C++";

  ClassList += ", CIR" + Name + "Lowering\n";
}
} // namespace

void clang::EmitCIRBuiltinsLowering(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Lowering of ClangIR builtins to LLVM IR builtins", OS);
  for (const auto *Builtin :
       Records.getAllDerivedDefinitions("LLVMLoweringInfo")) {
    if (!Builtin->getValueAsString("llvmOp").empty())
      GenerateLowering(Builtin);
  }

  OS << "#ifdef GET_BUILTIN_LOWERING_CLASSES\n"
     << ClassDefinitions << "\n#undef GET_BUILTIN_LOWERING_CLASSES\n#endif\n";
  OS << "#ifdef GET_BUILTIN_LOWERING_LIST\n"
     << ClassList << "\n#undef GET_BUILTIN_LOWERING_LIST\n#endif\n";
}
