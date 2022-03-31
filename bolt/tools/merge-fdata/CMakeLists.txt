set(LLVM_LINK_COMPONENTS Support)

add_llvm_tool(merge-fdata
  merge-fdata.cpp

  DEPENDS
  intrinsics_gen
)
set_target_properties(merge-fdata PROPERTIES FOLDER "BOLT")

# Emit relocations for BOLT meta test (bolt/test/runtime/meta-merge-fdata.test)
if (BOLT_INCLUDE_TESTS AND UNIX AND NOT APPLE)
  target_link_options(merge-fdata PRIVATE LINKER:--emit-relocs)
endif()
