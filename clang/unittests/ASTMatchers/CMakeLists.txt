set(LLVM_LINK_COMPONENTS
  Support
  )

# By default MSVC has a 2^16 limit on the number of sections in an object file,
# and this needs more than that.
if (MSVC)
  set_source_files_properties(InternalASTMatchersTest.cpp PROPERTIES COMPILE_FLAGS /bigobj)
  set_source_files_properties(NodeMatchersTest.cpp PROPERTIES COMPILE_FLAGS /bigobj)
  set_source_files_properties(NarrowingMatchersTest.cpp PROPERTIES COMPILE_FLAGS /bigobj)
  set_source_files_properties(ASTTraversalMatchersTest.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_clang_unittest(ASTMatchersTests
  ASTMatchersInternalTest.cpp
  ASTMatchersNodeTest.cpp
  ASTMatchersNarrowingTest.cpp
  ASTMatchersTraversalTest.cpp
  )

target_link_libraries(ASTMatchersTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )

add_subdirectory(Dynamic)
