add_custom_target(FlangUnitTests)
set_target_properties(FlangUnitTests PROPERTIES FOLDER "Flang Unit Tests")

function(add_flang_unittest test_dirname)
  add_unittest(FlangUnitTests ${test_dirname} ${ARGN})
endfunction()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_definitions("-Wno-suggest-override")
endif()

add_subdirectory(Optimizer)
add_subdirectory(Decimal)
add_subdirectory(Evaluate)
add_subdirectory(Runtime)
add_subdirectory(Lower)
