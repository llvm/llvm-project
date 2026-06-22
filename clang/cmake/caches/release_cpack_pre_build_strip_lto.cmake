file(GLOB files ${CPACK_TEMPORARY_INSTALL_DIRECTORY}/lib/*.a)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(strip_command
    ${CPACK_TEMPORARY_INSTALL_DIRECTORY}/bin/llvm-bitcode-strip)
  set(strip_args -r)
else()
  set(strip_command ${CPACK_TEMPORARY_INSTALL_DIRECTORY}/bin/llvm-strip)
  set(strip_args --no-strip-all -R .llvm.lto)
endif()

foreach(file ${files})
  execute_process(COMMAND ${strip_command} ${strip_args} ${file} -o ${file})
endforeach()
