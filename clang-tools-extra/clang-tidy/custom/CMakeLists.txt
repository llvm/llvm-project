if(CLANG_TIDY_ENABLE_QUERY_BASED_CUSTOM_CHECKS)
  set(LLVM_LINK_COMPONENTS
    FrontendOpenMP
    support
  )

  add_clang_library(clangTidyCustomModule STATIC
    CustomTidyModule.cpp
    QueryCheck.cpp

    LINK_LIBS
    clangQuery
    clangTidy
    clangTidyBugproneModule
    clangTidyMiscModule
    clangTidyModernizeModule
    clangTidyPerformanceModule
    clangTidyReadabilityModule
    clangTidyUtils

    DEPENDS
    omp_gen
    ClangDriverOptions
  )

  clang_target_link_libraries(clangTidyCustomModule
    PRIVATE
    clangAnalysis
    clangAST
    clangASTMatchers
    clangBasic
    clangDynamicASTMatchers
    clangFrontend
    clangLex
    clangSerialization
    clangTooling
  )
endif()
