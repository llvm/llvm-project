if (COMPILER_RT_BUILD_SANITIZERS)
  set(SANITIZER_HEADERS
    sanitizer/allocator_interface.h
    sanitizer/asan_interface.h
    sanitizer/common_interface_defs.h
    sanitizer/coverage_interface.h
    sanitizer/dfsan_interface.h
    sanitizer/esan_interface.h
    sanitizer/hwasan_interface.h
    sanitizer/linux_syscall_hooks.h
    sanitizer/lsan_interface.h
    sanitizer/msan_interface.h
    sanitizer/netbsd_syscall_hooks.h
    sanitizer/scudo_interface.h
    sanitizer/tsan_interface.h
    sanitizer/tsan_interface_atomic.h
    )
endif(COMPILER_RT_BUILD_SANITIZERS)

if (COMPILER_RT_BUILD_XRAY)
  set(XRAY_HEADERS
    xray/xray_interface.h
    xray/xray_log_interface.h
    )
endif(COMPILER_RT_BUILD_XRAY)

set(COMPILER_RT_HEADERS
  ${SANITIZER_HEADERS}
  ${XRAY_HEADERS})

set(output_dir ${COMPILER_RT_OUTPUT_DIR}/include)

# Copy compiler-rt headers to the build tree.
set(out_files)
foreach( f ${COMPILER_RT_HEADERS} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying compiler-rt's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )

add_custom_target(compiler-rt-headers ALL DEPENDS ${out_files})
add_dependencies(compiler-rt compiler-rt-headers)
set_target_properties(compiler-rt-headers PROPERTIES FOLDER "Compiler-RT Misc")

# Install sanitizer headers.
install(FILES ${SANITIZER_HEADERS}
  COMPONENT compiler-rt-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${COMPILER_RT_INSTALL_PATH}/include/sanitizer)
# Install xray headers.
install(FILES ${XRAY_HEADERS}
  COMPONENT compiler-rt-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${COMPILER_RT_INSTALL_PATH}/include/xray)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDEs.
  add_custom_target(install-compiler-rt-headers
                    DEPENDS compiler-rt-headers
                    COMMAND "${CMAKE_COMMAND}"
                            -DCMAKE_INSTALL_COMPONENT="compiler-rt-headers"
                            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                    USES_TERMINAL)
  add_custom_target(install-compiler-rt-headers-stripped
                    DEPENDS compiler-rt-headers
                    COMMAND "${CMAKE_COMMAND}"
                            -DCMAKE_INSTALL_COMPONENT="compiler-rt-headers"
                            -DCMAKE_INSTALL_DO_STRIP=1
                            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                    USES_TERMINAL)
endif()
