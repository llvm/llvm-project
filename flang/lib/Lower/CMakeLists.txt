get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_flang_library(FortranLower
  Allocatable.cpp
  Bridge.cpp
  CallInterface.cpp
  ConvertArrayConstructor.cpp
  ConvertCall.cpp
  ConvertConstant.cpp
  ConvertExprToHLFIR.cpp
  ConvertProcedureDesignator.cpp
  ConvertType.cpp
  ConvertVariable.cpp
  CUDA.cpp
  CustomIntrinsicCall.cpp
  HlfirIntrinsics.cpp
  HostAssociations.cpp
  IO.cpp
  LoweringOptions.cpp
  Mangler.cpp
  MultiImageFortran.cpp
  OpenACC.cpp
  OpenMP/Atomic.cpp
  OpenMP/ClauseProcessor.cpp
  OpenMP/Clauses.cpp
  OpenMP/DataSharingProcessor.cpp
  OpenMP/Decomposer.cpp
  OpenMP/OpenMP.cpp
  OpenMP/Utils.cpp
  PFTBuilder.cpp
  Runtime.cpp
  Support/PrivateReductionUtils.cpp
  Support/ReductionProcessor.cpp
  Support/Utils.cpp
  SymbolMap.cpp
  VectorSubscripts.cpp

  DISABLE_PCH_REUSE
  PRECOMPILE_HEADERS
  [["flang/Lower/SymbolMap.h"]]
  [["flang/Lower/AbstractConverter.h"]]
  [["flang/Lower/CallInterface.h"]]
  [["flang/Lower/BoxAnalyzer.h"]]
  [["flang/Lower/PFTBuilder.h"]]
  [["flang/Lower/DirectivesCommon.h"]]
  
  DEPENDS
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRTransforms
  HLFIRDialect
  MIFDialect

  LINK_LIBS
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRDialectSupport
  FIRBuilder
  FIROpenACCSupport
  FIRSupport
  FIRTransforms
  HLFIRDialect
  FortranSupport
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranUtils
  MIFDialect

  LINK_COMPONENTS
  FrontendOpenMP
  Support

  MLIR_DEPS
  ${dialect_libs}
  ${extension_libs}

  MLIR_LIBS
  ${dialect_libs}
  ${extension_libs}
  MLIRAffineToStandard
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSCFToControlFlow
)
