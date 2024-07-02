# Get the linker version on Darwin
function(get_darwin_linker_version variable)
  set(LINK_VERSION)
  set(LD_V_OUTPUT)
  execute_process(
    COMMAND sh -c "${CMAKE_LINKER} -v 2>&1 | head -1"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE LD_V_OUTPUT
    )
  if (HAD_ERROR)
    message(FATAL_ERROR "${CMAKE_LINKER} failed with status ${HAD_ERROR}")
  endif()
  if ("${LD_V_OUTPUT}" MATCHES ".*ld64-([0-9.]+).*")
    string(REGEX REPLACE ".*ld64-([0-9.]+).*" "\\1" LINK_VERSION ${LD_V_OUTPUT})
  elseif ("${LD_V_OUTPUT}" MATCHES "[^0-9]*([0-9.]+).*")
    string(REGEX REPLACE "[^0-9]*([0-9.]+).*" "\\1" LINK_VERSION ${LD_V_OUTPUT})
  endif()
  set(${variable} ${LINK_VERSION} PARENT_SCOPE)
endfunction()
