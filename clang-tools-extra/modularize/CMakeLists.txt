set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(modularize
  Modularize.cpp
  ModuleAssistant.cpp
  ModularizeUtilities.cpp
  CoverageChecker.cpp
  PreprocessorTracker.cpp
  )

target_link_libraries(modularize
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangLex
  clangTooling
  )

install(TARGETS modularize
        RUNTIME DESTINATION bin
        COMPONENT clang-extras)
