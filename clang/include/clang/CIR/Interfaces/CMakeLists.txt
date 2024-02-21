# This replicates part of the add_mlir_interface cmake function from MLIR that
# cannot be used here. This happens because it expects to be run inside MLIR
# directory which is not the case for CIR (and also FIR, both have similar
# workarounds).

# Declare a dialect in the include directory
function(add_clang_mlir_attr_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-attr-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-attr-interface-defs)
  add_public_tablegen_target(MLIRCIR${interface}IncGen)
  add_dependencies(mlir-generic-headers MLIRCIR${interface}IncGen)
endfunction()

function(add_clang_mlir_op_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(MLIR${interface}IncGen)
  add_dependencies(mlir-generic-headers MLIR${interface}IncGen)
endfunction()

function(add_clang_mlir_type_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-type-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-type-interface-defs)
  add_public_tablegen_target(MLIR${interface}IncGen)
  add_dependencies(mlir-generic-headers MLIR${interface}IncGen)
endfunction()

add_clang_mlir_attr_interface(ASTAttrInterfaces)
add_clang_mlir_op_interface(CIROpInterfaces)
add_clang_mlir_op_interface(CIRLoopOpInterface)
add_clang_mlir_type_interface(CIRFPTypeInterface)
