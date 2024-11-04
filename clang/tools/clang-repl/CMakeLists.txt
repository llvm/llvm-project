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
