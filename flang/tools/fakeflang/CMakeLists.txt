set(LLVM_LINK_COMPONENTS
  Support
)
set(FLANG_BUILD_TOOLS OFF) # Do not install
add_flang_tool(fakeflang
  fakeflang.cpp
)

# Phony target that ensures fakeflang can be called to emulate compiler probes.
# Clang is required to handle the preprocessing.
add_custom_target(flang-lazy)
add_dependencies(flang-lazy fakeflang clang)
