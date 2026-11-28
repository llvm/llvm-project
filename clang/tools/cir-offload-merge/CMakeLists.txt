include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(CIROffloadMergeOptsTableGen)

add_clang_tool(cir-offload-merge
  cir-offload-merge.cpp

  DEPENDS
  CIROffloadMergeOptsTableGen
)

target_link_libraries(cir-offload-merge
  PRIVATE
  CIRRegisterAllDialects
  MLIRCIR
  MLIRIR
  MLIRParser
  MLIRSupport
)
