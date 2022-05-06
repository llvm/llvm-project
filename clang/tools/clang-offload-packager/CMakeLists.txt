set(LLVM_LINK_COMPONENTS 
  ${LLVM_TARGETS_TO_BUILD}
  Object
  Support)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_executable(clang-offload-packager
  ClangOffloadPackager.cpp

  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_LINKER_WRAPPER_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-offload-packager)

target_link_libraries(clang-offload-packager
  PRIVATE
  ${CLANG_LINKER_WRAPPER_LIB_DEPS}
  )

install(TARGETS clang-offload-packager RUNTIME DESTINATION bin)
