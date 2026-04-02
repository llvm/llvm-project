set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Support
  TargetParser
  TextAPI
  TextAPIBinaryReader
  Option
  )

set(LLVM_TARGET_DEFINITIONS InstallAPIOpts.td)
tablegen(LLVM InstallAPIOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(InstallAPIDriverOptions)

# Options.cpp uses "using namespace llvm" which makes llvm::Value ambiguous
# with other types when combined in unity builds.
set_source_files_properties(
  Options.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_tool(clang-installapi
  ClangInstallAPI.cpp
  Options.cpp

  DEPENDS
  InstallAPIDriverOptions
  GENERATE_DRIVER
  )

clang_target_link_libraries(clang-installapi
  PRIVATE
  clangAST
  clangInstallAPI
  clangBasic
  clangOptions
  clangDriver
  clangFrontend
  clangTooling
  clangSerialization
  )

