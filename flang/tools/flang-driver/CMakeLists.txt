# Infrastructure to build flang driver entry point. Flang driver depends on
# LLVM libraries.

# Set your project compile flags.
link_directories(${LLVM_LIBRARY_DIR})

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  TargetParser
)

option(FLANG_PLUGIN_SUPPORT "Build Flang with plugin support." ON)

# Enable support for plugins, which need access to symbols from flang-new
if(FLANG_PLUGIN_SUPPORT)
  set(export_symbols EXPORT_SYMBOLS_FOR_PLUGINS)
endif()

add_flang_tool(flang-new
  driver.cpp
  fc1_main.cpp

  ${export_symbols}
)

target_link_libraries(flang-new
  PRIVATE
  flangFrontend
  flangFrontendTool
)

clang_target_link_libraries(flang-new
  PRIVATE
  clangDriver
  clangBasic
)

install(TARGETS flang-new DESTINATION "${CMAKE_INSTALL_BINDIR}")
