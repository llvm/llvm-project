if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

set(error_to_string_platform_dep "")
if(TARGET libc.src.__support.StringUtil.tables.${LIBC_TARGET_OS}.error_table)
  set(error_to_string_platform_dep 
  libc.src.__support.StringUtil.tables.${LIBC_TARGET_OS}.error_table)
endif()

add_header_library(
  error_table
  HDRS
    error_table.h  
    stdc_error_table.h  
    posix_error_table.h  
  DEPENDS
    ${error_to_string_platform_dep}
)

set(signal_to_string_platform_dep "")
if(TARGET libc.src.__support.StringUtil.tables.${LIBC_TARGET_OS}.signal_table)
  set(signal_to_string_platform_dep 
  libc.src.__support.StringUtil.tables.${LIBC_TARGET_OS}.signal_table)
endif()

add_header_library(
  signal_table
  HDRS
    signal_table.h  
    stdc_signal_table.h  
    posix_signal_table.h  
  DEPENDS
    ${signal_to_string_platform_dep}
)
