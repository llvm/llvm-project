set(LLVM_TARGET_DEFINITIONS HLFIROpBase.td)
mlir_tablegen(HLFIRTypes.h.inc -gen-typedef-decls -typedefs-dialect=hlfir)
mlir_tablegen(HLFIRTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hlfir)
mlir_tablegen(HLFIRDialect.h.inc -gen-dialect-decls -dialect=hlfir)
mlir_tablegen(HLFIRDialect.cpp.inc -gen-dialect-defs -dialect=hlfir)
mlir_tablegen(HLFIRAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=hlfir)
mlir_tablegen(HLFIRAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hlfir)

set(LLVM_TARGET_DEFINITIONS HLFIROps.td)
mlir_tablegen(HLFIROps.h.inc -gen-op-decls)
mlir_tablegen(HLFIROps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HLFIR)

add_public_tablegen_target(HLFIROpsIncGen)
