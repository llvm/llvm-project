set(LLVM_LINK_COMPONENTS
   core
   native
   MC
   Option
   OrcJit
   OrcDebugging
   OrcShared
   OrcTargetProcess
   Support
   Target
   TargetParser
  )

add_clang_library(clangInterpreter
  DeviceOffload.cpp
  CodeCompletion.cpp
  IncrementalExecutor.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  InterpreterUtils.cpp
  Value.cpp

  DEPENDS
  intrinsics_gen
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSema
  clangSerialization
  )

if ((MINGW OR CYGWIN) AND BUILD_SHARED_LIBS)
  # The DLLs are supposed to export all symbols (except for ones that are
  # explicitly hidden). Normally, this is what happens anyway, but if there
  # are symbols that are marked explicitly as dllexport, we'd only export them
  # and nothing else. The Interpreter contains a few cases of such dllexports
  # (for symbols that need to be exported even from standalone exe files);
  # therefore, add --export-all-symbols to make sure we export all symbols
  # despite potential dllexports.
  target_link_options(clangInterpreter PRIVATE LINKER:--export-all-symbols)
endif()
