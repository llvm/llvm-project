include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

add_clang_tool(cir-offload-merge
  cir-offload-merge.cpp
)

target_link_libraries(cir-offload-merge
  PRIVATE
  CIRRegisterAllDialects
  MLIRCIR
  MLIRIR
  MLIRParser
  MLIRSupport
)
