include_directories(SYSTEM ${CLANG_CIR_MLIR_INCLUDE_DIRS})

add_distinct_clang_unittest(CIRUnitTests
  CallOpTest.cpp
  CIRDiagnosticHandlerTest.cpp
  ControlFlowTest.cpp
  FenvOpTest.cpp
  GetFloatingPointTypeTest.cpp
  IntTypeABIAlignTest.cpp
  PointerLikeTest.cpp
  RecordMemberKindTest.cpp
  RecordTypeMetadataTest.cpp
  RecordTypeVerifyTest.cpp
  UnionTypeSizeTest.cpp
  VectorTypeABIAlignTest.cpp
  LLVM_COMPONENTS
  Core

  LINK_LIBS
  clangBasic
  clangCIRFrontendAction
  MLIRCIR
  CIROpenACCSupport
  MLIRIR
  MLIROpenACCDialect
  MLIRParser
  )
