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

# The clang-repl binary can get huge with static linking in debug mode.
# Some 32-bit targets use PLT slots with limited branch range by default and we
# start to exceed this limit, e.g. when linking for arm-linux-gnueabihf with
# gold. This flag tells the linker to build a PLT for the full address range.
# Linkers without this flag are assumed to support proper PLTs by default.
set(flag_long_plt "-Wl,--long-plt")
llvm_check_linker_flag(CXX ${flag_long_plt} HAVE_LINKER_FLAG_LONG_PLT)
if(HAVE_LINKER_FLAG_LONG_PLT)
  target_link_options(clang-repl PRIVATE ${flag_long_plt})
endif()
