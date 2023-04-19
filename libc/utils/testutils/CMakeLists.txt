if(CMAKE_HOST_UNIX)
  set(EFFile ExecuteFunctionUnix.cpp)
  set(FDReaderFile FDReaderUnix.cpp)
endif()

add_library(
  libc_test_utils
  ${EFFile}
  ExecuteFunction.h
  ${FDReaderFile}
  FDReader.h
  RoundingModeUtils.cpp
)
