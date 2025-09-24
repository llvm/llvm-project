set(LLVM_LINK_COMPONENTS Support)

add_clang_tool(offload-arch OffloadArch.cpp NVPTXArch.cpp AMDGPUArchByKFD.cpp
               AMDGPUArchByHIP.cpp LevelZeroArch.cpp)

find_path(OFFLOAD_ARCH_LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h)
if (OFFLOAD_ARCH_LEVEL_ZERO_INCLUDE_DIR)
  target_include_directories(offload-arch PRIVATE ${OFFLOAD_ARCH_LEVEL_ZERO_INCLUDE_DIR})
  target_compile_definitions(offload-arch PRIVATE HAVE_LEVEL_ZERO_HEADERS)
endif()

add_clang_symlink(amdgpu-arch offload-arch)
add_clang_symlink(nvptx-arch offload-arch)
add_clang_symlink(intelgpu-arch offload-arch)

target_link_libraries(offload-arch PRIVATE clangBasic)
