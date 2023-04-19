if(CMAKE_HOST_UNIX)
  set(EFFile ExecuteFunctionUnix.cpp)
endif()

add_library(
  libc_test_utils
  ${EFFile}
  ExecuteFunction.h
  RoundingModeUtils.cpp
)
