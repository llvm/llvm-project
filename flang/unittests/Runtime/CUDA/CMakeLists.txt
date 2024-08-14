if (FLANG_CUF_RUNTIME)

add_flang_unittest(FlangCufRuntimeTests
  AllocatorCUF.cpp
)

target_link_libraries(FlangCufRuntimeTests
  PRIVATE
  CufRuntime
  FortranRuntime
)

target_include_directories(FlangCufRuntimeTests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

endif()
