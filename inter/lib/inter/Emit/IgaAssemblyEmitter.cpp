#include "inter/Emit/Emit.h"

#include "iga.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

mlir::LogicalResult inter::emitAssembly(mlir::ModuleOp moduleOp,
                                        llvm::raw_ostream &output) {
  llvm::SmallVector<char> binary;
  llvm::raw_svector_ostream binaryOutput(binary);
  if (mlir::failed(emitGedBinary(moduleOp, binaryOutput)))
    return mlir::failure();

  iga_context_options_t contextOptions = IGA_CONTEXT_OPTIONS_INIT(IGA_XE2);
  iga_context_t context = nullptr;
  iga_status_t status = iga_context_create(&contextOptions, &context);
  if (status != IGA_SUCCESS)
    return moduleOp.emitError("IGA context creation failed: ")
               << iga_status_to_string(status),
           mlir::failure();

  iga_disassemble_options_t options = IGA_DISASSEMBLE_OPTIONS_INIT();
  char *assembly = nullptr;
  status = iga_context_disassemble(context, &options, binary.data(),
                                   static_cast<uint32_t>(binary.size()),
                                   nullptr, nullptr, &assembly);
  if (status != IGA_SUCCESS) {
    mlir::InFlightDiagnostic diagnostic =
        moduleOp.emitError("IGA disassembly failed: ")
        << iga_status_to_string(status);
    const iga_diagnostic_t *errors = nullptr;
    uint32_t errorCount = 0;
    if (iga_context_get_errors(context, &errors, &errorCount) == IGA_SUCCESS)
      for (uint32_t index = 0; index < errorCount; ++index)
        diagnostic.attachNote()
            << "byte " << errors[index].offset << ": " << errors[index].message;
    iga_context_release(context);
    return mlir::failure();
  }

  const iga_diagnostic_t *warnings = nullptr;
  uint32_t warningCount = 0;
  if (iga_context_get_warnings(context, &warnings, &warningCount) ==
      IGA_SUCCESS)
    for (uint32_t index = 0; index < warningCount; ++index)
      moduleOp.emitWarning()
          << "IGA disassembly warning at byte " << warnings[index].offset
          << ": " << warnings[index].message;

  output << assembly;
  iga_context_release(context);
  return mlir::success();
}
