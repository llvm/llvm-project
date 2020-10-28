# Infrastructure to build flang driver entry point. Flang driver depends on
# LLVM libraries.

# Set your project compile flags.
link_directories(${LLVM_LIBRARY_DIR})

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
)

add_flang_tool(flang-new
  driver.cpp
  fc1_main.cpp
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
  clangFrontend
)

install(TARGETS flang-new DESTINATION bin)
