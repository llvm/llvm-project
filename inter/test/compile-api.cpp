#include "inter/Compiler/Compiler.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>

static std::unique_ptr<llvm::Module> parseModule(llvm::LLVMContext &context,
                                                 llvm::StringRef triple) {
  std::string ir = R"(
target datalayout = "e-i64:64-G1"
target triple = ")";
  ir += triple;
  ir += R"("
define spir_kernel void @kernel(ptr addrspace(1) %output) {
  ret void
}
)";
  llvm::SMDiagnostic diagnostic;
  return llvm::parseAssemblyString(ir, diagnostic, context);
}

static uint16_t read16(llvm::ArrayRef<char> bytes, size_t offset) {
  return static_cast<uint8_t>(bytes[offset]) |
         static_cast<uint16_t>(static_cast<uint8_t>(bytes[offset + 1])) << 8;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: inter-compile-api-test <pipeline-library>\n";
    return 1;
  }

  inter::CompilerOptions options;
  options.transformLibraryPath = argv[1];
  llvm::LLVMContext context;
  llvm::SmallVector<char> zebin;
  llvm::raw_svector_ostream output(zebin);
  if (llvm::Error error = inter::compileLLVMModule(
          parseModule(context, "spir64-unknown-unknown"), output, llvm::errs(),
          options)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  if (zebin.size() < 64 || read16(zebin, 16) != 0xFF12 ||
      read16(zebin, 18) != llvm::ELF::EM_INTELGT) {
    llvm::errs() << "compiler API emitted an invalid Zebin header\n";
    return 1;
  }
  llvm::outs() << "zebin: " << zebin.size() << " bytes\n";

  options.output = inter::CompilationOutput::ged;
  llvm::LLVMContext gedContext;
  llvm::SmallVector<char> ged;
  llvm::raw_svector_ostream gedOutput(ged);
  if (llvm::Error error = inter::compileLLVMModule(
          parseModule(gedContext, "spir64-unknown-unknown"), gedOutput,
          llvm::errs(), options)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  if (ged.empty() || ged.size() % 16 != 0) {
    llvm::errs() << "compiler API emitted malformed GED bytes\n";
    return 1;
  }
  llvm::outs() << "ged: " << ged.size() << " bytes\n";

  options.output = inter::CompilationOutput::assembly;
  llvm::LLVMContext assemblyContext;
  std::string assembly;
  llvm::raw_string_ostream assemblyOutput(assembly);
  if (llvm::Error error = inter::compileLLVMModule(
          parseModule(assemblyContext, "spir64-unknown-unknown"),
          assemblyOutput, llvm::errs(), options)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  if (assembly.empty()) {
    llvm::errs() << "compiler API emitted empty assembly\n";
    return 1;
  }
  llvm::outs() << "assembly: present\n";

  options.output = inter::CompilationOutput::none;
  llvm::LLVMContext validationContext;
  llvm::SmallVector<char> validation;
  llvm::raw_svector_ostream validationOutput(validation);
  if (llvm::Error error = inter::compileLLVMModule(
          parseModule(validationContext, "spir64-unknown-unknown"),
          validationOutput, llvm::errs(), options)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  if (!validation.empty()) {
    llvm::errs() << "validation-only compilation emitted output\n";
    return 1;
  }
  llvm::outs() << "validation: passed\n";

  llvm::LLVMContext invalidContext;
  llvm::SmallVector<char> ignored;
  llvm::raw_svector_ostream ignoredOutput(ignored);
  options.output = inter::CompilationOutput::zebin;
  llvm::Error error = inter::compileLLVMModule(
      parseModule(invalidContext, "x86_64-unknown-linux-gnu"), ignoredOutput,
      llvm::errs(), options);
  if (!error) {
    llvm::errs() << "compiler API accepted an invalid target triple\n";
    return 1;
  }
  llvm::outs() << "diagnostic: " << llvm::toString(std::move(error));
  return 0;
}
