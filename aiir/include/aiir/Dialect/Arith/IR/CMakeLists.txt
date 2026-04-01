set(LLVM_TARGET_DEFINITIONS ArithOps.td)
aiir_tablegen(ArithOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(ArithOpsEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(ArithOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=arith)
aiir_tablegen(ArithOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=arith)
add_aiir_dialect(ArithOps arith)
add_aiir_doc(ArithOps ArithOps Dialects/ -gen-dialect-doc)
add_aiir_interface(ArithOpsInterfaces)
