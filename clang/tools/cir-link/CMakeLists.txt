get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_clang_tool(cir-link
  cir-link.cpp
)

clang_target_link_libraries(cir-link
  PRIVATE
  clangCIR
  MLIRCIR
  MLIRLinkLib
  MLIRSupport
)

target_include_directories(cir-link
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/../mlir/include
  ${CMAKE_BINARY_DIR}/tools/mlir/include
)
