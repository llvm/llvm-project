if (NOT LIBCXX_INSTALL_SUPPORT_HEADERS)
  set(LIBCXX_SUPPORT_HEADER_PATTERN PATTERN "support" EXCLUDE)
endif()

set(LIBCXX_HEADER_PATTERN
  PATTERN "*"
  PATTERN "CMakeLists.txt" EXCLUDE
  PATTERN ".svn" EXCLUDE
  PATTERN "__config_site.in" EXCLUDE
  ${LIBCXX_SUPPORT_HEADER_PATTERN}
  )

file(COPY .
  DESTINATION "${CMAKE_BINARY_DIR}/include/c++/v1"
  FILES_MATCHING
  ${LIBCXX_HEADER_PATTERN}
  )

if (LIBCXX_INSTALL_HEADERS)
  install(DIRECTORY .
    DESTINATION include/c++/v1
    COMPONENT cxx-headers
    FILES_MATCHING
    ${LIBCXX_HEADER_PATTERN}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  )

  if (LIBCXX_NEEDS_SITE_CONFIG)
    set(UNIX_CAT cat)
    if (WIN32)
      set(UNIX_CAT type)
    endif()
    # Generate and install a custom __config header. The new header is created
    # by  prepending __config_site to the current __config header.
    add_custom_command(OUTPUT ${LIBCXX_BINARY_DIR}/__generated_config
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBCXX_BINARY_DIR}/__config_site ${LIBCXX_BINARY_DIR}/__generated_config
      COMMAND ${UNIX_CAT} ${LIBCXX_SOURCE_DIR}/include/__config >> ${LIBCXX_BINARY_DIR}/__generated_config
      DEPENDS ${LIBCXX_SOURCE_DIR}/include/__config
              ${LIBCXX_BINARY_DIR}/__config_site
    )
    # Add a target that executes the generation commands.
    add_custom_target(generate_config_header ALL
      DEPENDS ${LIBCXX_BINARY_DIR}/__generated_config)
    set(generated_config_deps generate_config_header)
    # Install the generated header as __config.
    install(FILES ${LIBCXX_BINARY_DIR}/__generated_config
      DESTINATION include/c++/v1
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
      RENAME __config
      COMPONENT cxx-headers)
  endif()

  if (NOT CMAKE_CONFIGURATION_TYPES)
    # this target is just needed as a placeholder for the distribution target
    add_custom_target(cxx-headers)
    add_custom_target(install-cxx-headers
                      DEPENDS cxx-headers ${generated_config_deps}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=cxx-headers
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")

    add_custom_target(libcxx-headers)
    add_custom_target(install-libcxx-headers DEPENDS install-cxx-headers)
  endif()

endif()
