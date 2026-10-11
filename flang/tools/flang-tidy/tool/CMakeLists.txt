set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  FrontendOpenACC
  FrontendOpenMP
  Support
  Option
  TargetParser
  )

set(LLVM_OPTIONAL_SOURCES FlangTidyMain.cpp FlangTidyToolMain.cpp)

add_flang_library(flangTidyMain STATIC
  FlangTidyMain.cpp

  LINK_LIBS
  FortranSupport
  FortranEvaluate
  FortranParser
  FortranSemantics
  LLVMSupport
  flangTidy
  ${ALL_FLANG_TIDY_CHECKS}
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
  flangTidy
  flangTidyMain
  ${ALL_FLANG_TIDY_CHECKS}
  )

if(FLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(flang-tidy)
endif()
