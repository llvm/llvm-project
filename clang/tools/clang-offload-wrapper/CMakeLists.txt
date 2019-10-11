set(LLVM_LINK_COMPONENTS BitWriter Core Support TransformUtils)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_tool(clang-offload-wrapper
  ClangOffloadWrapper.cpp

  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_OFFLOAD_WRAPPER_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-offload-wrapper)

clang_target_link_libraries(clang-offload-wrapper
  PRIVATE
  ${CLANG_OFFLOAD_WRAPPER_LIB_DEPS}
  )
