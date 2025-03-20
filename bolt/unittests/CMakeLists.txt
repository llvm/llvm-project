add_custom_target(BoltUnitTests)
set_target_properties(BoltUnitTests PROPERTIES FOLDER "BOLT/Tests")

function(add_bolt_unittest test_dirname)
  add_unittest(BoltUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Core)
add_subdirectory(Profile)
