set(LLVM_LINK_COMPONENTS Support)

add_bolt_tool(merge-fdata
  merge-fdata.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  intrinsics_gen
)

add_dependencies(bolt merge-fdata)

# Emit relocations for BOLT meta test (bolt/test/runtime/meta-merge-fdata.test)
if (BOLT_INCLUDE_TESTS AND UNIX AND NOT APPLE)
  target_link_options(merge-fdata PRIVATE LINKER:--emit-relocs)
endif()
