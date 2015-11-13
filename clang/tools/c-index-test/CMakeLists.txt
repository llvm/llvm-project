add_clang_executable(c-index-test
  c-index-test.c
  )

if(NOT MSVC)
  set_property(
    SOURCE c-index-test.c
    PROPERTY COMPILE_FLAGS "-std=gnu89"
    )
endif()

if (LLVM_BUILD_STATIC)
  target_link_libraries(c-index-test
    libclang_static
  )
else()
  target_link_libraries(c-index-test
    libclang
  )
endif()

set_target_properties(c-index-test
  PROPERTIES
  LINKER_LANGUAGE CXX)

# If libxml2 is available, make it available for c-index-test.
if (CLANG_HAVE_LIBXML)
  include_directories(SYSTEM ${LIBXML2_INCLUDE_DIR})
  target_link_libraries(c-index-test ${LIBXML2_LIBRARIES})
endif()

if(INTERNAL_INSTALL_PREFIX)
  set(INSTALL_DESTINATION "${INTERNAL_INSTALL_PREFIX}/bin")
else()
  set(INSTALL_DESTINATION bin)
endif()

install(TARGETS c-index-test
  RUNTIME DESTINATION "${INSTALL_DESTINATION}"
  COMPONENT c-index-test)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_custom_target(install-c-index-test
    DEPENDS c-index-test
    COMMAND "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=c-index-test
            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
endif()
