get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_flang_library(FortranLower
  Allocatable.cpp
  Bridge.cpp
  CallInterface.cpp
  Coarray.cpp
  ComponentPath.cpp
  ConvertArrayConstructor.cpp
  ConvertCall.cpp
  ConvertConstant.cpp
  ConvertExpr.cpp
  ConvertExprToHLFIR.cpp
  ConvertProcedureDesignator.cpp
  ConvertType.cpp
  ConvertVariable.cpp
  CustomIntrinsicCall.cpp
  DumpEvaluateExpr.cpp
  HlfirIntrinsics.cpp
  HostAssociations.cpp
  IO.cpp
  IterationSpace.cpp
  LoweringOptions.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  Runtime.cpp
  SymbolMap.cpp
  VectorSubscripts.cpp
  
  DEPENDS
  FIRDialect
  FIRTransforms
  HLFIRDialect
  ${dialect_libs}
  ${extension_libs}

  LINK_LIBS
  FIRDialect
  FIRDialectSupport
  FIRBuilder
  FIRSupport
  FIRTransforms
  HLFIRDialect
  ${dialect_libs}
  ${extension_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSCFToControlFlow

  LINK_COMPONENTS
  Support
)
