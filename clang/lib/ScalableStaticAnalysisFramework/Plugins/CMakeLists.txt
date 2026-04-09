if(CLANG_PLUGIN_SUPPORT AND LLVM_ENABLE_PLUGINS)
  # Plugins must never bring LLVM or Clang libraries in statically.
  # clang-ssaf-analyzer already loads clangScalableStaticAnalysisFrameworkCore
  # and LLVM into the process; a second static copy would produce duplicate
  # llvm::Registry instances with separate global state, breaking registration.
  #
  # Instead: build each plugin as a MODULE with PLUGIN_TOOL pointing to the
  # tool that will load it, and pull only include paths (not link deps) from
  # the libraries whose headers are needed. All symbols are resolved from the
  # tool's address space when the plugin is loaded via --load.
  add_subdirectory(ExamplePlugin)
endif()
