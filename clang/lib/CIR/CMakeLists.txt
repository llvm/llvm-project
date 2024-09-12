include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

# Report use of deprecated APIs as errors.
# TODO: Consider adding `/we4996` for MSVC when upstream MLIR resolves
#       https://github.com/llvm/llvm-project/issues/65255.
if (NOT MSVC)
  add_compile_options("-Werror=deprecated-declarations")
endif()

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Lowering)
add_subdirectory(Interfaces)
