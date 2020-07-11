check_cxx_compiler_flag("-Werror -Wc99-extensions" HAS_WC99_EXTENSIONS_FLAG)
if (HAS_WC99_EXTENSIONS_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-extensions")
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  Bridge.cpp
  CallInterface.cpp
  CharacterExpr.cpp
  CharacterRuntime.cpp
  Coarray.cpp
  ComplexExpr.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  IntrinsicCall.cpp
  IO.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  Runtime.cpp

  DEPENDS
  FIROptimizer
  ${dialect_libs}
  omp_gen
  acc_gen

  LINK_LIBS
  FIROptimizer
  ${dialect_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandard

  LINK_COMPONENTS
  Support
)
