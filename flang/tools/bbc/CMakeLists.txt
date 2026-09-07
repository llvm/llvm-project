set(LLVM_LINK_COMPONENTS
  Passes
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  TargetParser
)

add_flang_tool(bbc bbc.cpp
  DEPENDS
  FIROptCodeGenPassIncGen
)

llvm_update_compile_flags(bbc)
target_link_libraries(bbc PRIVATE
  CUFAttrs
  CUFDialect
  FIRDialect
  FIRDialectSupport
  FIROpenACCSupport
  FIROpenMPSupport
  FIRSupport
  FIRTransforms
  FIRBuilder
  HLFIRDialect
  HLFIRTransforms
  flangFrontend
  flangPasses
  FlangOpenMPTransforms
  FortranSupport
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
  MIFDialect
)

mlir_target_link_libraries(bbc PRIVATE
  ${FLANG_MLIR_DIALECT_LIBS}
  ${FLANG_MLIR_EXTENSION_LIBS}
  MLIRAffineToStandard
  MLIRSCFToControlFlow
)
