if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  message(STATUS "Skipping the loader for ${LIBC_TARGET_ARCHITECTURE}")
  return()
endif()

# The loader relocates itself before anything else runs, so its own code must
# not need a relocation applied to reach a global or make a call. Hidden
# visibility and no PLT are what keep the compiler from emitting one.
set(LIBC_LOADER_COMPILE_OPTIONS
  -fvisibility=hidden
  -fno-plt
  -fno-stack-protector
  -fno-exceptions
  -fno-rtti
  -ffreestanding
  -fno-builtin
  -ftls-model=initial-exec
)

add_library(libc-loader-objects OBJECT
  ${CMAKE_CURRENT_SOURCE_DIR}/loader_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE}/start.cpp
)
target_include_directories(libc-loader-objects
  SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
target_include_directories(libc-loader-objects PRIVATE ${LIBC_SOURCE_DIR})
target_compile_options(libc-loader-objects PRIVATE
  ${LIBC_LOADER_COMPILE_OPTIONS} -DLIBC_COPT_PUBLIC_PACKAGING)
set_target_properties(libc-loader-objects
  PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(libc-loader-objects libc-headers)

add_library(libc-loader SHARED $<TARGET_OBJECTS:libc-loader-objects>)
add_dependencies(libc-loader libc-loader-objects)
set_target_properties(libc-loader
  PROPERTIES
    OUTPUT_NAME ld-llvm-libc-${LIBC_TARGET_ARCHITECTURE}.so.1
    PREFIX ""
    SUFFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR})

# The loader is its own interpreter and depends on nothing, so it links
# against no library at all and its entry point is the stub rather than a
# shared object's usual one.
target_link_options(libc-loader PRIVATE
  -nostdlib
  -Wl,-e,_start
  -Wl,--no-undefined
  -Wl,-z,now
)
if(LIBC_COMPILER_RT_BUILTINS)
  target_link_libraries(libc-loader PRIVATE ${LIBC_COMPILER_RT_BUILTINS})
endif()

install(TARGETS libc-loader
  LIBRARY DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
  COMPONENT libc)
