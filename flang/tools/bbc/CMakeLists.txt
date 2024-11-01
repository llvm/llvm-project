set(LLVM_LINK_COMPONENTS
Passes
)

add_flang_tool(bbc bbc.cpp
DEPENDS
FIROptCodeGenPassIncGen
)

llvm_update_compile_flags(bbc)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(bbc PRIVATE
FIRDialect
FIRSupport
FIRTransforms
FIRBuilder
HLFIRDialect
HLFIRTransforms
${dialect_libs}
MLIRAffineToStandard
MLIRSCFToControlFlow
FortranCommon
FortranParser
FortranEvaluate
FortranSemantics
FortranLower
)
