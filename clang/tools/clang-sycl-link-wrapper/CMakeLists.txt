set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  )

set(LLVM_TARGET_DEFINITIONS SYCLLinkOpts.td)
tablegen(LLVM SYCLLinkOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(SYCLLinkWrapperOpts)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen SYCLLinkWrapperOpts)
endif()

add_clang_tool(clang-sycl-link-wrapper
  ClangSYCLLinkWrapper.cpp

  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_SYCL_LINK_WRAPPER_LIB_DEPS
  clangBasic
  )

target_link_libraries(clang-sycl-link-wrapper
  PRIVATE
  ${CLANG_SYCL_LINK_WRAPPER_LIB_DEPS}
  )
