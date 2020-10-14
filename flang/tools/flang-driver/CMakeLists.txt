# Infrastructure to build flang driver entry point. Flang driver depends on
# LLVM libraries.

# Set your project compile flags.
link_directories(${LLVM_LIBRARY_DIR})

add_flang_tool(flang-new
  driver.cpp
  fc1_main.cpp

  LINK_COMPONENTS

  ${LLVM_COMMON_COMPONENTS}
  Support
  Target
  Option
)

# Link against LLVM and Clang libraries
target_link_libraries(flang-new
  PRIVATE
  flangFrontend
  flangFrontendTool
  clangDriver
  clangBasic
  clangFrontend
)

install(TARGETS flang-new DESTINATION bin)
