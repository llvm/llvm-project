set(LLVM_LINK_COMPONENTS Support)

add_clang_tool(offload-arch OffloadArch.cpp NVPTXArch.cpp AMDGPUArchByKFD.cpp
               AMDGPUArchByHIP.cpp LevelZeroArch.cpp)

# Legacy binary names.
add_clang_symlink(amdgpu-arch offload-arch)
add_clang_symlink(nvptx-arch offload-arch)

target_link_libraries(offload-arch PRIVATE clangBasic)
