set(AIIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../aiir/include )
set(AIIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/aiir/include)
include_directories(SYSTEM ${AIIR_INCLUDE_DIR})
include_directories(${AIIR_TABLEGEN_OUTPUT_DIR})

add_distinct_clang_unittest(CIRUnitTests
  PointerLikeTest.cpp
  LLVM_COMPONENTS
  Core

  LINK_LIBS
  AIIRCIR
  CIROpenACCSupport
  AIIRIR
  AIIROpenACCDialect
  )
