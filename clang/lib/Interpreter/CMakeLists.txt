set(LLVM_LINK_COMPONENTS
   core
   native
   MC
   Option
   Support
   Target
   TargetParser
   TransformUtils
   )

if (EMSCRIPTEN)
  if (NOT TARGET lldWasm)
    find_package(LLD REQUIRED CONFIG)
  endif()
  set(EXECUTOR_SOURCES Wasm.cpp)
  set(EXECUTOR_LIBS lldWasm lldCommon)
else()
  list(APPEND LLVM_LINK_COMPONENTS
    OrcJit
    OrcDebugging
    OrcShared
    OrcTargetProcess
    )
  set(EXECUTOR_SOURCES IncrementalExecutor.cpp OrcIncrementalExecutor.cpp)
endif()

add_clang_library(clangInterpreter
  DeviceOffload.cpp
  CodeCompletion.cpp
  IncrementalAction.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  InterpreterValuePrinter.cpp
  InterpreterUtils.cpp
  Value.cpp
  InterpreterValuePrinter.cpp
  ${EXECUTOR_SOURCES}
  PARTIAL_SOURCES_INTENDED

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
  clangOptions
  clangParse
  clangSema
  clangSerialization
  ${EXECUTOR_LIBS}
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
