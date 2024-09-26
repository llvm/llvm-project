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

add_flang_tool(flang
  driver.cpp
  fc1_main.cpp
)

target_link_libraries(flang
  PRIVATE
  flangFrontend
  flangFrontendTool
)

clang_target_link_libraries(flang
  PRIVATE
  clangDriver
  clangBasic
)

option(FLANG_PLUGIN_SUPPORT "Build Flang with plugin support." ON)

# Enable support for plugins, which need access to symbols from flang
if(FLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(flang)
endif()

install(TARGETS flang DESTINATION "${CMAKE_INSTALL_BINDIR}")

# Keep "flang-new" as a symlink for backwards compatiblity. Remove once "flang"
# is a widely adopted name.
add_flang_symlink(flang-new flang)
