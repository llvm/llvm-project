set(files
  __cxxabi_config.h
  cxxabi.h
  )

add_library(cxxabi-headers INTERFACE)
target_include_directories(cxxabi-headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if (LIBCXXABI_INSTALL_HEADERS)
  foreach(file ${files})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file}
      DESTINATION ${LIBCXXABI_INSTALL_INCLUDE_DIR}/${dir}
      COMPONENT cxxabi-headers
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
  endforeach()

  add_custom_target(install-cxxabi-headers
                    DEPENDS cxxabi-headers
                    COMMAND "${CMAKE_COMMAND}"
                            -DCMAKE_INSTALL_COMPONENT=cxxabi-headers
                            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
  # Stripping is a no-op for headers
  add_custom_target(install-cxxabi-headers-stripped DEPENDS install-cxxabi-headers)
endif()
