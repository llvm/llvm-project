get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

# GCC, unlike clang, issues a warning when one virtual function is overridden
# in a derived class but one or more other virtual functions with the same
# name and different signature from a base class are not overridden. This
# leads to many warnings in the MLIR and ClangIR code when using the
# OpenConversionPattern<>::matchAndRewrite() function in the ordinary way.
# The "hiding" behavior is what we want, so we're just disabling the warning
# here.
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual")
endif()

add_clang_tool(cir-opt
  cir-opt.cpp
)

clang_target_link_libraries(cir-opt
  PRIVATE
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRCIR
  MLIRCIRTransforms
)

target_link_libraries(cir-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRIR
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformUtils
)
