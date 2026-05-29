# The sanitizer ignorelists are default special-case lists that the Clang
# driver passes to the compiler (via -fsanitize-system-ignorelist) when the
# corresponding sanitizer is enabled. They are consumed by the compiler rather
# than linked into any runtime library, so they are built by their own target
# that does not depend on the sanitizer runtimes and can be built on its own.
add_compiler_rt_component(sanitizer-ignorelists)

set(SANITIZER_IGNORELIST_FILES
  asan_ignorelist.txt
  cfi_ignorelist.txt
  hwasan_ignorelist.txt
  msan_ignorelist.txt
  )

set(sanitizer_ignorelist_outputs)
foreach(file_name ${SANITIZER_IGNORELIST_FILES})
  set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}")
  set(dst_file "${COMPILER_RT_OUTPUT_DIR}/share/${file_name}")
  add_custom_command(OUTPUT ${dst_file}
    DEPENDS ${src_file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_file} ${dst_file}
    COMMENT "Copying ${file_name}...")
  list(APPEND sanitizer_ignorelist_outputs ${dst_file})
endforeach()

add_custom_target(sanitizer-ignorelist-files DEPENDS ${sanitizer_ignorelist_outputs})
set_target_properties(sanitizer-ignorelist-files PROPERTIES FOLDER "Compiler-RT/Resources")
add_dependencies(sanitizer-ignorelists sanitizer-ignorelist-files)

# Install in the Clang resource directory. Each sanitizer runtime's install
# target (install-asan, install-hwasan, ...) depends on
# install-sanitizer-ignorelists, so installing or distributing any sanitizer
# also installs the ignorelists. These install targets are created
# unconditionally (like compiler-rt's other parent install targets) so the
# runtime install targets can depend on them under multi-config generators too.
install(FILES ${SANITIZER_IGNORELIST_FILES}
  DESTINATION ${COMPILER_RT_INSTALL_DATA_DIR}
  COMPONENT sanitizer-ignorelists)

add_custom_target(install-sanitizer-ignorelists
  DEPENDS sanitizer-ignorelists
  COMMAND "${CMAKE_COMMAND}"
          -DCMAKE_INSTALL_COMPONENT=sanitizer-ignorelists
          -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
add_custom_target(install-sanitizer-ignorelists-stripped
  DEPENDS sanitizer-ignorelists
  COMMAND "${CMAKE_COMMAND}"
          -DCMAKE_INSTALL_COMPONENT=sanitizer-ignorelists
          -DCMAKE_INSTALL_DO_STRIP=1
          -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
set_target_properties(install-sanitizer-ignorelists PROPERTIES
  FOLDER "Compiler-RT/Installation")
set_target_properties(install-sanitizer-ignorelists-stripped PROPERTIES
  FOLDER "Compiler-RT/Installation")
add_dependencies(install-compiler-rt install-sanitizer-ignorelists)
add_dependencies(install-compiler-rt-stripped install-sanitizer-ignorelists-stripped)
