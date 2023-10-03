set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_library(FlangRTFortranEvaluateTesting
  testing.cpp
)
if (LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs Support)
endif()
target_link_libraries(FlangRTFortranEvaluateTesting
    ${llvm_libs})

add_flang_rt_nongtest_unittest(reshape
  FlangRTFortranEvaluateTesting
  flang-rt
)

add_flang_rt_nongtest_unittest(ISO-Fortran-binding
  FlangRTFortranEvaluateTesting
  flang-rt
)
