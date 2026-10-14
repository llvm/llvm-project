set(LLVM_LINK_COMPONENTS Support)

add_clang_tool(offload-arch OffloadArch.cpp)

# Legacy binary names.
add_clang_symlink(amdgpu-arch offload-arch)
add_clang_symlink(nvptx-arch offload-arch)

add_subdirectory(lib)
target_link_libraries(offload-arch PRIVATE clangBasic clangOffloadArch)
