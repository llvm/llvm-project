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

clang_target_link_libraries(modularize
  PRIVATE
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )

install(TARGETS modularize
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clang-extras)
