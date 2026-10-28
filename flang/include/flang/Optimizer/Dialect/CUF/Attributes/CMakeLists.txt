set(LLVM_TARGET_DEFINITIONS CUFAttr.td)
mlir_tablegen(CUFEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(CUFEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(CUFAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(CUFAttr.cpp.inc -gen-attrdef-defs)

add_flang_tablegen_target(CUFAttrsIncGen)
