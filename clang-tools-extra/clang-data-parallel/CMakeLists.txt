add_llvm_tool(clang-data-parallel DataParallelCheck.cpp )

target_link_libraries(clang-data-parallel PRIVATE clangAST clangBasic clangFrontend clangTooling clangToolingCore LLVMFrontendOpenMP LLVMSupport )

set_target_properties(clang-data-parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )