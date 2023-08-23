set(PACKAGE_PREFIX ${CMAKE_INSTALL_LIBDIR}/cmake/AMDDeviceLibs)

# Generate the build-tree package.
# We know the absolute path to the build tree, so we leave
# AMD_DEVICE_LIBS_PREFIX_CODE blank and include absolute paths in the target
# imports in AMD_DEVICE_LIBS_TARGET_CODE.
foreach(target ${AMDGCN_LIB_LIST})
  get_target_property(target_path ${target} OUTPUT_NAME)
  set(AMD_DEVICE_LIBS_TARGET_CODE "${AMD_DEVICE_LIBS_TARGET_CODE}
add_library(${target} STATIC IMPORTED)
set_target_properties(${target} PROPERTIES
  IMPORTED_LOCATION \"${target_path}\")")
endforeach()
configure_file(AMDDeviceLibsConfig.cmake.in
  ${PACKAGE_PREFIX}/AMDDeviceLibsConfig.cmake
  @ONLY)


set(install_path_suffix "amdgcn/bitcode")

# Generate the install-tree package.
# We do not know the absolute path to the intall tree until we are installed,
# so we calculate it dynamically in AMD_DEVICE_LIBS_PREFIX_CODE and use
# relative paths in the target imports in AMD_DEVICE_LIBS_TARGET_CODE.
set(AMD_DEVICE_LIBS_PREFIX_CODE "
# Derive absolute install prefix from config file path.
get_filename_component(AMD_DEVICE_LIBS_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
string(REGEX REPLACE "/" ";" count "${PACKAGE_PREFIX}")
foreach(p ${count})
  set(AMD_DEVICE_LIBS_PREFIX_CODE "${AMD_DEVICE_LIBS_PREFIX_CODE}
get_filename_component(AMD_DEVICE_LIBS_PREFIX \"\${AMD_DEVICE_LIBS_PREFIX}\" PATH)")
endforeach()
set(AMD_DEVICE_LIBS_TARGET_CODE)
foreach(target ${AMDGCN_LIB_LIST})
  get_target_property(target_name ${target} ARCHIVE_OUTPUT_NAME)
  get_target_property(target_prefix ${target} PREFIX)
  get_target_property(target_suffix ${target} SUFFIX)
  set(AMD_DEVICE_LIBS_TARGET_CODE "${AMD_DEVICE_LIBS_TARGET_CODE}
add_library(${target} STATIC IMPORTED)
set_target_properties(${target} PROPERTIES
  IMPORTED_LOCATION \"\${AMD_DEVICE_LIBS_PREFIX}/${install_path_suffix}/${target_prefix}${target_name}${target_suffix}\")")
endforeach()
configure_file(AMDDeviceLibsConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/AMDDeviceLibsConfig.cmake.install
  @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AMDDeviceLibsConfig.cmake.install
  DESTINATION ${PACKAGE_PREFIX}
  COMPONENT device-libs
  RENAME AMDDeviceLibsConfig.cmake)
