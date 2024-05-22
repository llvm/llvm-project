set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LineEditor
  Option
  OrcJIT
  Support
  )

add_clang_tool(clang-repl
  ClangRepl.cpp
  )

if(MSVC)
  set_target_properties(clang-repl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)

  # RTTI/C++ symbols
  set(clang_repl_exports ${clang_repl_exports} ??_7type_info@@6B@
    ?__type_info_root_node@@3U__type_info_node@@A
    ?nothrow@std@@3Unothrow_t@1@B
  )

  # Compiler added symbols for static variables. NOT for VStudio < 2015
  set(clang_repl_exports ${clang_repl_exports} _Init_thread_abort _Init_thread_epoch
    _Init_thread_footer _Init_thread_header _tls_index
  )

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # new/delete variants needed when linking to static msvc runtime (esp. Debug)
    set(clang_repl_exports ${clang_repl_exports}
      ??2@YAPEAX_K@Z
      ??3@YAXPEAX@Z
      ??_U@YAPEAX_K@Z
      ??_V@YAXPEAX@Z
      ??3@YAXPEAX_K@Z
    )
  else()
    set(clang_repl_exports ${clang_repl_exports}
      ??2@YAPAXI@Z
      ??3@YAXPAX@Z
      ??3@YAXPAXI@Z
      ??_U@YAPAXI@Z
      ??_V@YAXPAX@Z
      ??_V@YAXPAXI@Z
    )
  endif()

  # List to '/EXPORT:sym0 /EXPORT:sym1 /EXPORT:sym2 ...'
  foreach(sym ${clang_repl_exports})
    set(clang_repl_link_str "${clang_repl_link_str} /EXPORT:${sym}")
  endforeach(sym ${clang_repl_exports})

  set_property(TARGET clang-repl APPEND_STRING PROPERTY LINK_FLAGS ${clang_repl_link_str})

endif(MSVC)

clang_target_link_libraries(clang-repl PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangInterpreter
  )

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(clang-repl)
endif()

string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" system_processor)
if(system_processor MATCHES "ARM")
  set(FLAG_LONG_PLT "-Wl,--long-plt")
  llvm_check_linker_flag(CXX ${FLAG_LONG_PLT} LINKER_HAS_FLAG_LONG_PLT)
  # Linkers without this flag are assumed to have long PLTs by default
  if(LINKER_HAS_FLAG_LONG_PLT)
    target_link_options(clang-repl PRIVATE ${FLAG_LONG_PLT})
  endif()
endif()
