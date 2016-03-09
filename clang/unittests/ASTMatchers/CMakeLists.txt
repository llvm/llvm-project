set(LLVM_LINK_COMPONENTS
  Support
  )

# By default MSVC has a 2^16 limit on the number of sections in an object file,
# and this needs more than that.
if (MSVC)
  set_source_files_properties(ASTMatchersTest.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_clang_unittest(ASTMatchersTests
  ASTMatchersTest.cpp)

target_link_libraries(ASTMatchersTests
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  )

add_subdirectory(Dynamic)
