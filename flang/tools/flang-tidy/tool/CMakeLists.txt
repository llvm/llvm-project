set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  FrontendOpenACC
  FrontendOpenMP
  Support
  Option
  TargetParser
  )

if(FLANG_PLUGIN_SUPPORT)
  set(support_plugins SUPPORT_PLUGINS)
endif()

add_flang_tool(flang-tidy
  FlangTidyToolMain.cpp

  ${support_plugins}
  )

target_link_libraries(flang-tidy
  PRIVATE
  ${ALL_FLANG_TIDY_CHECKS}
  )

if(FLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(flang-tidy)
endif()
