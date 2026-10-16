set(LLVM_LINK_COMPONENTS Support)

add_bolt_tool(merge-fdata
  merge-fdata.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  intrinsics_gen
)

add_dependencies(bolt merge-fdata)

if (BOLT_INCLUDE_TESTS)
  # Compile another binary using --emit-relocs for BOLT meta test in
  # bolt/test/runtime/meta-merge-fdata.test.
  set(EXCLUDE_FROM_ALL ON)
  add_bolt_executable(merge-fdata-for-test
    merge-fdata.cpp

    DISABLE_LLVM_LINK_LLVM_DYLIB

    DEPENDS
    intrinsics_gen
  )
  if (UNIX AND NOT APPLE)
    target_link_options(merge-fdata-for-test PRIVATE LINKER:--emit-relocs)
  endif()
endif()
