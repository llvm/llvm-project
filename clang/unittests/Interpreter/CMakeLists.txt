set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  MC
  OrcJIT
  Support
  TargetParser
  )

add_clang_unittest(ClangReplInterpreterTests
  IncrementalCompilerBuilderTest.cpp
  IncrementalProcessingTest.cpp
  InterpreterTest.cpp
  InterpreterExtensionsTest.cpp
  CodeCompletionTest.cpp
  )
target_link_libraries(ClangReplInterpreterTests PUBLIC
  clangAST
  clangBasic
  clangInterpreter
  clangFrontend
  clangSema
  LLVMTestingSupport
  )

# Exceptions on Windows are not yet supported.
if(NOT WIN32)
  add_subdirectory(ExceptionTests)
endif()

export_executable_symbols(ClangReplInterpreterTests)

if(MSVC)
  set_target_properties(ClangReplInterpreterTests PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)

  # RTTI/C++ symbols
  set(ClangReplInterpreterTests_exports ${ClangReplInterpreterTests_exports} ??_7type_info@@6B@
    ?__type_info_root_node@@3U__type_info_node@@A
    ?nothrow@std@@3Unothrow_t@1@B
  )

  # Compiler added symbols for static variables. NOT for VStudio < 2015
  set(ClangReplInterpreterTests_exports ${ClangReplInterpreterTests_exports} _Init_thread_abort _Init_thread_epoch
    _Init_thread_footer _Init_thread_header _tls_index
  )

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # new/delete variants needed when linking to static msvc runtime (esp. Debug)
    set(ClangReplInterpreterTests_exports ${ClangReplInterpreterTests_exports}
      ??2@YAPEAX_K@Z
      ??3@YAXPEAX@Z
      ??_U@YAPEAX_K@Z
      ??_V@YAXPEAX@Z
      ??3@YAXPEAX_K@Z
    )
  else()
    set(ClangReplInterpreterTests_exports ${ClangReplInterpreterTests_exports}
      ??2@YAPAXI@Z
      ??3@YAXPAX@Z
      ??3@YAXPAXI@Z
      ??_U@YAPAXI@Z
      ??_V@YAXPAX@Z
      ??_V@YAXPAXI@Z
    )
  endif()

  # List to '/EXPORT:sym0 /EXPORT:sym1 /EXPORT:sym2 ...'
  foreach(sym ${ClangReplInterpreterTests_exports})
    set(ClangReplInterpreterTests_link_str "${ClangReplInterpreterTests_link_str} /EXPORT:${sym}")
  endforeach(sym ${ClangReplInterpreterTests_exports})

  set_property(TARGET ClangReplInterpreterTests APPEND_STRING PROPERTY LINK_FLAGS ${ClangReplInterpreterTests_link_str})

endif(MSVC)
