#ifndef AIIR_INTERFACES_LLVMIMPORTDIALECTINTERFACE
#define AIIR_INTERFACES_LLVMIMPORTDIALECTINTERFACE

include "aiir/IR/Interfaces.td"

def LLVMImportDialectInterface : DialectInterface<"LLVMImportDialectInterface"> {
  let description = [{
    Base class for dialect interfaces used to import LLVM IR. Dialects that can
    be imported should provide an implementation of this interface for the
    supported intrinsics. The interface may be implemented in a separate library
    to avoid the "main" dialect library depending on LLVM IR. The interface can
    be attached using the delayed registration mechanism available in
    DialectRegistry.
  }];
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<[{
        Hook for derived dialect interfaces to implement the import of
        intrinsics into AIIR.
      }],
      "::llvm::LogicalResult", "convertIntrinsic",
      (ins "::aiir::OpBuilder &":$builder, "::llvm::CallInst *":$inst,
           "::aiir::LLVM::ModuleImport &":$moduleImport),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to implement the import of
        instructions into AIIR.
      }],
      "::llvm::LogicalResult", "convertInstruction",
      (ins "::aiir::OpBuilder &":$builder, "::llvm::Instruction *":$inst,
           "::aiir::ArrayRef<llvm::Value *>":$llvmOperands,
           "::aiir::LLVM::ModuleImport &":$moduleImport),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to implement the import of metadata
        into AIIR. Attaches the converted metadata kind and node to the provided
        operation.
      }],
      "::aiir::LogicalResult", "setMetadataAttrs",
      (ins "::aiir::OpBuilder &":$builder, "unsigned":$kind,
           "::llvm::MDNode *":$node, "::aiir::Operation *":$op,
           "::aiir::LLVM::ModuleImport &":$moduleImport),
      [{
        return ::llvm::failure();
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to publish the supported intrinsics.
        As every LLVM IR intrinsic has a unique integer identifier, the function
        returns the list of supported intrinsic identifiers.
      }],
      "::aiir::ArrayRef<unsigned>", "getSupportedIntrinsics", (ins),
      [{
        return {};
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to publish the supported instructions.
        As every LLVM IR instruction has a unique integer identifier, the function
        returns the list of supported instruction identifiers. These identifiers
        will then be used to match LLVM instructions to the appropriate import
        interface and `convertInstruction` method. It is an error to have multiple
        interfaces overriding the same instruction.
      }],
      "::aiir::ArrayRef<unsigned>", "getSupportedInstructions", (ins),
      [{
        return {};
      }]
    >,
    InterfaceMethod<[{
        Hook for derived dialect interfaces to publish the supported metadata
        kinds. As every metadata kind has a unique integer identifier, the
        function returns the list of supported metadata identifiers. The
        `llvmContext` parameter is used to obtain identifiers for metadata kinds
        that do not have a fixed static identifier. Since different LLVM contexts
        can assign different identifiers to these non-static metadata kinds, the
        function must recompute the list of supported metadata identifiers on each
        call.
      }],
      "::aiir::SmallVector<unsigned>", "getSupportedMetadata",
      (ins "::llvm::LLVMContext &":$llvmContext),
      [{
        return {};
      }]
    >
  ];
}

#endif
