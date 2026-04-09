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

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Interfaces)
add_subdirectory(Lowering)
