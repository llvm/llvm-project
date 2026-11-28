# HAS_WERROR_GLOBAL_CTORS and LLVM_HAS_NOGLOBAL_CTOR_MUTEX are cached by
# llvm/lib/Support/CMakeLists.txt. They are unset in a standalone build, which
# skips the check.
if (HAS_WERROR_GLOBAL_CTORS AND LLVM_HAS_NOGLOBAL_CTOR_MUTEX)
  set(CLANG_LIB_WERROR_GLOBAL_CTORS ON)
  string(APPEND CMAKE_CXX_FLAGS " -Werror=global-constructors")
endif()

add_subdirectory(Headers)
add_subdirectory(Basic)
add_subdirectory(APINotes)
add_subdirectory(Lex)
add_subdirectory(Parse)
add_subdirectory(AST)
add_subdirectory(ASTMatchers)
add_subdirectory(CrossTU)
add_subdirectory(Sema)
add_subdirectory(CodeGenUtils)
add_subdirectory(CodeGen)
add_subdirectory(Analysis)
add_subdirectory(Edit)
add_subdirectory(ExtractAPI)
add_subdirectory(Rewrite)
add_subdirectory(Driver)
add_subdirectory(Options)
add_subdirectory(Serialization)
add_subdirectory(Frontend)
add_subdirectory(FrontendTool)
add_subdirectory(Tooling)
add_subdirectory(DependencyScanning)
add_subdirectory(DirectoryWatcher)
add_subdirectory(Index)
add_subdirectory(IndexSerialization)
add_subdirectory(InstallAPI)
add_subdirectory(ScalableStaticAnalysis)
add_subdirectory(StaticAnalyzer)
add_subdirectory(Format)
if(CLANG_INCLUDE_TESTS)
  add_subdirectory(Testing)
endif()
add_subdirectory(Interpreter)
add_subdirectory(Support)
add_subdirectory(UnifiedSymbolResolution)

if(CLANG_ENABLE_CIR)
  add_subdirectory(CIR)
endif()

# Libraries that still have static initializers. Remove a library once it has
# none, so that it cannot regain them.
if (CLANG_LIB_WERROR_GLOBAL_CTORS)
  foreach(lib
      SSAFExamplePlugin
      clangBasic
      clangDriver
      clangFormat
      clangFrontendTool
      clangInstallAPI
      clangInterpreter
      clangScalableStaticAnalysisAnalyses
      clangScalableStaticAnalysisCore
      clangScalableStaticAnalysisSourceTransformation
      clangSema
      clangSerialization
      clangStaticAnalyzerCheckers
      clangStaticAnalyzerCore
      clangStaticAnalyzerFrontend
      clangTooling
      clangToolingInclusions
      )
    if (TARGET obj.${lib})
      target_compile_options(obj.${lib} PRIVATE -Wno-global-constructors)
    elseif (TARGET ${lib})
      target_compile_options(${lib} PRIVATE -Wno-global-constructors)
    endif()
  endforeach()
endif()
