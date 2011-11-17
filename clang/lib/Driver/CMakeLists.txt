set(LLVM_USED_LIBS clangBasic clangAST clangParse)

add_clang_library(clangDriver
  Action.cpp
  Arg.cpp
  ArgList.cpp
  CC1Options.cpp
  CC1AsOptions.cpp
  Compilation.cpp
  Driver.cpp
  DriverOptions.cpp
  HostInfo.cpp
  Job.cpp
  Option.cpp
  OptTable.cpp
  Phases.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  Tools.cpp
  Types.cpp
  )

IF(MSVC)
  get_target_property(NON_ANSI_COMPILE_FLAGS clangDriver COMPILE_FLAGS)
  string(REPLACE /Za
    "" NON_ANSI_COMPILE_FLAGS
    ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(clangDriver PROPERTIES COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
ENDIF(MSVC)

add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)
