if (FLANG_CUF_RUNTIME)

add_flang_unittest(FlangCufRuntimeTests
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/CUDA/Allocatable.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/CUDA/AllocatorCUF.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/CUDA/Memory.cpp
)

if (BUILD_SHARED_LIBS)
  set(CUDA_RT_TARGET CUDA::cudart)
else()
  set(CUDA_RT_TARGET CUDA::cudart_static)
endif()

target_link_libraries(FlangCufRuntimeTests
  PRIVATE
  ${CUDA_RT_TARGET}
  flang_rt.cuda_${CUDAToolkit_VERSION_MAJOR}
  flang_rt.runtime
)

target_include_directories(FlangCufRuntimeTests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

endif()
