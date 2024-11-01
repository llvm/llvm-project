add_executable(amdhsa_loader Loader.cpp)
target_link_libraries(amdhsa_loader
  PRIVATE
  hsa-runtime64::hsa-runtime64
  gpu_loader
)
