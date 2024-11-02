add_flang_library(flangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  # This makes sure that the MLIR dependencies of flangFrontend (which are
  # transitively required here) are generated before this target is build.
  flangFrontend

  LINK_LIBS
  flangFrontend
  MLIRPass

  LINK_COMPONENTS
  Option
  Support
)

if(CLANG_LINK_CLANG_DYLIB)
  add_dependencies(flangFrontend clang-cpp)
else()
  add_dependencies(flangFrontendTool clangBasic)
endif()

clang_target_link_libraries(flangFrontendTool
  PRIVATE
  clangBasic
  clangDriver
)
