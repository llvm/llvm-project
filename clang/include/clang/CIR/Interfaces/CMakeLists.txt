# This replicates part of the add_aiir_interface cmake function from AIIR that
# cannot be used here. This happens because it expects to be run inside AIIR
# directory which is not the case for CIR (and also FIR, both have similar
# workarounds).

function(add_clang_aiir_attr_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  aiir_tablegen(${interface}.h.inc -gen-attr-interface-decls)
  aiir_tablegen(${interface}.cpp.inc -gen-attr-interface-defs)
  add_public_tablegen_target(AIIRCIR${interface}IncGen)
  add_dependencies(aiir-generic-headers AIIRCIR${interface}IncGen)
endfunction()

function(add_clang_aiir_op_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  aiir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  aiir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(AIIR${interface}IncGen)
  add_dependencies(aiir-generic-headers AIIR${interface}IncGen)
endfunction()

function(add_clang_aiir_type_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  aiir_tablegen(${interface}.h.inc -gen-type-interface-decls)
  aiir_tablegen(${interface}.cpp.inc -gen-type-interface-defs)
  add_public_tablegen_target(AIIR${interface}IncGen)
  add_dependencies(aiir-generic-headers AIIR${interface}IncGen)
endfunction()

add_clang_aiir_attr_interface(ASTAttrInterfaces)
add_clang_aiir_op_interface(CIROpInterfaces)
add_clang_aiir_op_interface(CIRLoopOpInterface)
add_clang_aiir_type_interface(CIRTypeInterfaces)
