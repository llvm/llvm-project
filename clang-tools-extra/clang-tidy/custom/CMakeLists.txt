if(CLANG_TIDY_ENABLE_QUERY_BASED_CUSTOM_CHECKS)
  set(LLVM_LINK_COMPONENTS
    support
  )

  add_clang_library(clangTidyCustomModule STATIC
    CustomTidyModule.cpp
    QueryCheck.cpp

    LINK_LIBS
    clangTidy
    clangTidyUtils

    DEPENDS
    ClangDriverOptions
  )

  clang_target_link_libraries(clangTidyCustomModule
    PRIVATE
    clangQuery
  )
endif()
