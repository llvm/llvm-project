set(LLVM_LINK_COMPONENTS BitWriter Core Object Support)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_executable(clang-nvlink-wrapper
  ClangNvlinkWrapper.cpp

  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_NVLINK_WRAPPER_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-nvlink-wrapper)

target_link_libraries(clang-nvlink-wrapper
  PRIVATE
  ${CLANG_NVLINK_WRAPPER_LIB_DEPS}
  )

install(TARGETS clang-nvlink-wrapper RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
