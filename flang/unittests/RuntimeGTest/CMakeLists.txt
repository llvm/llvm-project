add_flang_unittest(FlangRuntimeTests
  CharacterTest.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  FortranRuntime
)
