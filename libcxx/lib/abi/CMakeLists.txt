if (CMAKE_CXX_COMPILER_TARGET)
  set(triple "${CMAKE_CXX_COMPILER_TARGET}")
else()
  set(triple "${LLVM_DEFAULT_TARGET_TRIPLE}")
endif()
cxx_abi_identifier(abi_list_identifier
  "${triple}"
  "${LIBCXX_CXX_ABI}"
  "${LIBCXX_ABI_VERSION}"
  "${LIBCXX_ABI_UNSTABLE}"
  "${LIBCXX_ENABLE_EXCEPTIONS}"
  "${LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS}"
)

if (TARGET cxx_shared)
  set(abi_list_file "${CMAKE_CURRENT_SOURCE_DIR}/${abi_list_identifier}.abilist")
  set(abi_map_file "${CMAKE_CURRENT_SOURCE_DIR}/${abi_list_identifier}.linker.map")

  if (EXISTS "${abi_list_file}")
    add_custom_target(check-cxx-abilist
      "${Python3_EXECUTABLE}" "${LIBCXX_SOURCE_DIR}/utils/sym_diff.py"
          --only-stdlib-symbols
          --strict "${abi_list_file}"
          $<TARGET_FILE:cxx_shared>
      DEPENDS cxx_shared
      COMMENT "Testing libc++'s exported symbols against the ABI list")
  else()
    message(STATUS "ABI list file not generated for configuration ${abi_list_identifier}, `check-cxx-abilist` will not be available.")
  endif()

  set(generate_cxx_abilist_parameters "${LIBCXX_SOURCE_DIR}/utils/generate_abi_list.py"
            --output "${abi_list_file}"
            "$<TARGET_FILE:cxx_shared>")
  if (EXISTS ${abi_map_file})
    list(APPEND generate_cxx_abilist_parameters --mapfile "${abi_map_file}")
  endif()

  add_custom_target(generate-cxx-abilist
    COMMAND "${Python3_EXECUTABLE}" ${generate_cxx_abilist_parameters}
    DEPENDS cxx_shared
    COMMENT "Generating the ABI list file for configuration ${abi_list_identifier}")
else()
  message(STATUS "Not building a shared library for libc++ -- the ABI list targets will not be available.")
endif()
