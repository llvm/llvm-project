# Infrastructure to build flang driver entry point. Flang driver depends on
# LLVM libraries.

# Set your project compile flags.
link_directories(${LLVM_LIBRARY_DIR})

add_flang_tool(flang-new
  driver.cpp
  fc1_main.cpp
)

# Link against LLVM and Clang libraries
target_link_libraries(flang-new
  PRIVATE
  ${LLVM_COMMON_LIBS}
  flangFrontend
  flangFrontendTool
  clangDriver
  clangBasic
  LLVMSupport
  LLVMTarget
  LLVMOption
)

install(TARGETS flang-new DESTINATION bin)
