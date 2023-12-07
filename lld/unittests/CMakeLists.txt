set_target_properties(LLDUnitTests PROPERTIES FOLDER "lld tests")

function(add_lld_unittests test_dirname)
  add_unittest(LLDUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(AsLibAll)
add_subdirectory(AsLibELF)
