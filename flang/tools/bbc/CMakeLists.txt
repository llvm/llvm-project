set(LLVM_LINK_COMPONENTS
Passes
TargetParser
)

add_flang_tool(bbc bbc.cpp
DEPENDS
FIROptCodeGenPassIncGen
)

llvm_update_compile_flags(bbc)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

check_include_file(quadmath.h FOUND_QUADMATH_HEADER)
check_library_exists(quadmath sinq "" FOUND_QUADMATH_LIB)
if(FOUND_QUADMATH_HEADER AND FOUND_QUADMATH_LIB)
  set(QUADMATHLIB quadmath)
endif()

target_link_libraries(bbc PRIVATE
FIRDialect
FIRDialectSupport
FIRSupport
FIRTransforms
FIRBuilder
HLFIRDialect
HLFIRTransforms
${dialect_libs}
${extension_libs}
MLIRAffineToStandard
MLIRSCFToControlFlow
FortranCommon
FortranParser
FortranEvaluate
FortranSemantics
FortranLower
${QUADMATHLIB}
)
