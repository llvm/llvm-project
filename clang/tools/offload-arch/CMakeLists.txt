set(LLVM_LINK_COMPONENTS Option Support)

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(OffloadArchOptsTableGen)

add_clang_tool(offload-arch
  OffloadArch.cpp

  DEPENDS
  OffloadArchOptsTableGen
  )

# Legacy binary names.
add_clang_symlink(amdgpu-arch offload-arch)
add_clang_symlink(nvptx-arch offload-arch)

add_subdirectory(lib)
target_link_libraries(offload-arch PRIVATE clangBasic clangOffloadArch)
