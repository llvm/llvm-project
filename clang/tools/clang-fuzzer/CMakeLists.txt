if( LLVM_USE_SANITIZE_COVERAGE )
  set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD})

  add_clang_executable(clang-fuzzer
    EXCLUDE_FROM_ALL
    ClangFuzzer.cpp
    )

  target_link_libraries(clang-fuzzer
    ${CLANG_FORMAT_LIB_DEPS}
    clangAST
    clangBasic
    clangCodeGen
    clangDriver
    clangFrontend
    clangRewriteFrontend
    clangStaticAnalyzerFrontend
    clangTooling
    LLVMFuzzer
    )
endif()
