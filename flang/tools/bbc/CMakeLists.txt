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
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(bbc PRIVATE
CUFAttrs
CUFDialect
FIRDialect
FIRDialectSupport
FIRSupport
FIRTransforms
FIRBuilder
HLFIRDialect
HLFIRTransforms
FlangOpenMPTransforms
${dialect_libs}
${extension_libs}
MLIRAffineToStandard
MLIRSCFToControlFlow
FortranCommon
FortranParser
FortranEvaluate
FortranSemantics
FortranLower
)
