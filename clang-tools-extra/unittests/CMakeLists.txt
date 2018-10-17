add_custom_target(ExtraToolsUnitTests)
set_target_properties(ExtraToolsUnitTests PROPERTIES FOLDER "Extra Tools Unit Tests")

function(add_extra_unittest test_dirname)
  add_unittest(ExtraToolsUnitTests ${test_dirname} ${ARGN})
endfunction()

if(CLANG_BUILT_STANDALONE)
  # LLVMTestingSupport library is needed for clangd tests.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      AND NOT TARGET LLVMTestingSupport)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      lib/Testing/Support)
  endif()
endif()

add_subdirectory(change-namespace)
add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-doc)
add_subdirectory(clang-move)
add_subdirectory(clang-query)
add_subdirectory(clang-tidy)
add_subdirectory(clangd)
add_subdirectory(include-fixer)
