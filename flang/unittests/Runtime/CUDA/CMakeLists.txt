if (FLANG_CUF_RUNTIME)

add_flang_unittest(FlangCufRuntimeTests
  Allocatable.cpp
  AllocatorCUF.cpp
  Memory.cpp
)

if (BUILD_SHARED_LIBS)
  set(CUDA_RT_TARGET CUDA::cudart)
else()
  set(CUDA_RT_TARGET CUDA::cudart_static)
endif()

target_link_libraries(FlangCufRuntimeTests
  PRIVATE
  ${CUDA_RT_TARGET}
  CufRuntime_cuda_${CUDAToolkit_VERSION_MAJOR}
  FortranRuntime
)

target_include_directories(FlangCufRuntimeTests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

endif()
