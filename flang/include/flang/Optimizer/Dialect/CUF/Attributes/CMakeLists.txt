set(LLVM_TARGET_DEFINITIONS CUFAttr.td)
aiir_tablegen(CUFEnumAttr.h.inc -gen-enum-decls)
aiir_tablegen(CUFEnumAttr.cpp.inc -gen-enum-defs)
aiir_tablegen(CUFAttr.h.inc --gen-attrdef-decls)
aiir_tablegen(CUFAttr.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(CUFAttrsIncGen)
