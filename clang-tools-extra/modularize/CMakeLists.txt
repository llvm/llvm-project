set(LLVM_LINK_COMPONENTS
  Option
  Support
  TargetParser
  )

# ModuleAssistant.cpp defines a local Module class that conflicts with
# clang::Module when combined in unity builds.
set_source_files_properties(
  ModuleAssistant.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_tool(modularize
  Modularize.cpp
  ModuleAssistant.cpp
  ModularizeUtilities.cpp
  CoverageChecker.cpp
  PreprocessorTracker.cpp

  DEPENDS
  ClangDriverOptions
  )

clang_target_link_libraries(modularize
  PRIVATE
  clangAST
  clangBasic
  clangDriver
  clangOptions
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )

install(TARGETS modularize
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clang-extras)
