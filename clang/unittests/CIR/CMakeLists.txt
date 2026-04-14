set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include )
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
set(CLANG_CIR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/CIR)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(${CLANG_CIR_SRC_DIR})

add_distinct_clang_unittest(CIRUnitTests
  CIRABIRewriteContextTest.cpp
  PointerLikeTest.cpp
  LLVM_COMPONENTS
  Core

  LINK_LIBS
  MLIRABI
  MLIRCIR
  MLIRCIRTargetLowering
  CIROpenACCSupport
  MLIRIR
  MLIROpenACCDialect
  )
