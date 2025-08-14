if(EMSCRIPTEN)
set(LLVM_COMPONENTS_TO_LINK
  ""
  )
set(LLVM_LIBS_TO_LINK
  ""
  )
set(CLANG_LIBS_TO_LINK
  clangInterpreter
  )
else()
set(LLVM_COMPONENTS_TO_LINK
  ${LLVM_TARGETS_TO_BUILD}
  Core
  MC
  OrcJIT
  Support
  TargetParser
  )
set(LLVM_LIBS_TO_LINK
  LLVMTestingSupport
  )
set(CLANG_LIBS_TO_LINK
  clangAST
  clangBasic
  clangInterpreter
  clangFrontend
  clangSema
  )
endif()

add_distinct_clang_unittest(ClangReplInterpreterTests
  IncrementalCompilerBuilderTest.cpp
  IncrementalProcessingTest.cpp
  InterpreterTest.cpp
  InterpreterExtensionsTest.cpp
  CodeCompletionTest.cpp

  EXPORT_SYMBOLS

  CLANG_LIBS
  ${CLANG_LIBS_TO_LINK}

  LINK_LIBS
  ${LLVM_LIBS_TO_LINK}

  LLVM_COMPONENTS
  ${LLVM_COMPONENTS_TO_LINK}
  )

if(EMSCRIPTEN)
# Without the above you try to link to LLVMSupport twice, and end
# up with a duplicate symbol error when creating the main module
get_target_property(LINKED_LIBS ClangReplInterpreterTests LINK_LIBRARIES)
list(REMOVE_ITEM LINKED_LIBS LLVMSupport)
set_target_properties(ClangReplInterpreterTests PROPERTIES LINK_LIBRARIES "${LINKED_LIBS}")
target_link_options(ClangReplInterpreterTests
  PUBLIC "SHELL: -s MAIN_MODULE=1"
  PUBLIC "SHELL: -s ALLOW_MEMORY_GROWTH=1"
  PUBLIC "SHELL: -s STACK_SIZE=32mb"
  PUBLIC "SHELL: -s INITIAL_MEMORY=128mb"
  PUBLIC "SHELL: --emrun"
)
set_target_properties(ClangReplInterpreterTests PROPERTIES
  SUFFIX ".html"
)
endif()

# Exceptions on Windows are not yet supported.
if(NOT WIN32)
  add_subdirectory(ExceptionTests)
endif()

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
  list(TRANSFORM ClangReplInterpreterTests_exports PREPEND "LINKER:/EXPORT:")
  set_property(TARGET ClangReplInterpreterTests APPEND PROPERTY LINK_OPTIONS ${ClangReplInterpreterTests_exports})

endif(MSVC)
