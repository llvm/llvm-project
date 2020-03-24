set(LLVM_LINK_COMPONENTS BitWriter Core Object Support)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_executable(clang-unbundle-archive
  ClangUnbundleArchive.cpp

  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_UNBUNDLE_ARCHIVE_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-unbundle-archive)

target_link_libraries(clang-unbundle-archive
  PRIVATE
  ${CLANG_UNBUNDLE_ARCHIVE_LIB_DEPS}
  )

install(TARGETS clang-unbundle-archive RUNTIME DESTINATION bin)
