add_flang_tool(fir-opt fir-opt.cpp)
llvm_update_compile_flags(fir-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(fir-opt PRIVATE
  FIRDialect
  FIRSupport
  FIRTransforms
  FIRCodeGen
  ${dialect_libs}

  # TODO: these should be transitive dependencies from a target providing
  # "registerFIRPasses()"
  MLIRAffineToStandard
  MLIROptLib
)
