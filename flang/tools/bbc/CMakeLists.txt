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
get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY AIIR_EXTENSION_LIBS)

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

aiir_target_link_libraries(bbc PRIVATE
  ${dialect_libs}
  ${extension_libs}
  AIIRAffineToStandard
  AIIRSCFToControlFlow
)
