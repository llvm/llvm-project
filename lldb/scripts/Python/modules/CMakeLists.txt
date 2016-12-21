# Disable some warnings triggered by Python's headers.
check_cxx_compiler_flag("-Wno-macro-redefined"
                        CXX_SUPPORTS_NO_MACRO_REDEFINED)
if (CXX_SUPPORTS_NO_MACRO_REDEFINED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-macro-redefined")
endif ()

# build the Python readline suppression module only on Linux
if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux" OR "${CMAKE_SYSTEM_NAME}" STREQUAL "GNU" OR "${CMAKE_SYSTEM_NAME}" STREQUAL "kFreeBSD")
   add_subdirectory(readline)
endif()
