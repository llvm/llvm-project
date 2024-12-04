set(LLVM_LINK_COMPONENTS 
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  Object
  Support)

add_clang_tool(clang-offload-packager
  ClangOffloadPackager.cpp

  DEPENDS
  ${tablegen_deps}
  )

setup_host_tool(clang-offload-packager CLANG_OFFLOAD_PACKAGER_EXE
  clang-offload-packager_exe clang-offload-packager_target)

clang_target_link_libraries(clang-offload-packager
  PRIVATE
  clangBasic
  )
