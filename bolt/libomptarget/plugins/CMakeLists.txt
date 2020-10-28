##===----------------------------------------------------------------------===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Build plugins for the user system if available.
#
##===----------------------------------------------------------------------===##

# void build_generic_elf64(string tmachine, string tmachine_name, string tmachine_libname, string elf_machine_id);
# - build a plugin for an ELF based generic 64-bit target based on libffi.
# - tmachine: name of the machine processor as used in the cmake build system.
# - tmachine_name: name of the machine to be printed with the debug messages.
# - tmachine_libname: machine name to be appended to the plugin library name.
macro(build_generic_elf64 tmachine tmachine_name tmachine_libname tmachine_triple elf_machine_id)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "${tmachine}$")
  if(LIBOMPTARGET_DEP_LIBELF_FOUND)
    if(LIBOMPTARGET_DEP_LIBFFI_FOUND)
    
      libomptarget_say("Building ${tmachine_name} offloading plugin.")
    
      include_directories(${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR})
      include_directories(${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR})
      
      # Define macro to be used as prefix of the runtime messages for this target.
      add_definitions("-DTARGET_NAME=${tmachine_name}")
      
      # Define macro with the ELF ID for this target.
      add_definitions("-DTARGET_ELF_ID=${elf_machine_id}")
    
      add_library("bolt-omptarget.rtl.${tmachine_libname}" SHARED 
        ${CMAKE_CURRENT_SOURCE_DIR}/../generic-elf-64bit/src/rtl.cpp)
        
      # Install plugin under the lib destination folder.
      install(TARGETS "bolt-omptarget.rtl.${tmachine_libname}" 
        LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

      # Install aliases
     get_target_property(BOLT_LIBOMPTARGET_LIBRARY_DIR "bolt-omptarget.rtl.${tmachine_libname}" LIBRARY_OUTPUT_DIRECTORY)
      if(BOLT_LIBOMPTARGET_LIBRARY_DIR)
        add_custom_command(TARGET "bolt-omptarget.rtl.${tmachine_libname}" POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SHARED_LIBRARY_PREFIX}bolt-omptarget.rtl.${tmachine_libname}${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${CMAKE_SHARED_LIBRARY_PREFIX}omptarget.rtl.${tmachine_libname}${CMAKE_SHARED_LIBRARY_SUFFIX}
          WORKING_DIRECTORY ${BOLT_LIBOMPTARGET_LIBRARY_DIR}
        )
      endif()
      install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${CMAKE_SHARED_LIBRARY_PREFIX}bolt-omptarget.rtl.${tmachine_libname}${CMAKE_SHARED_LIBRARY_SUFFIX}\"
        \"${CMAKE_SHARED_LIBRARY_PREFIX}omptarget.rtl.${tmachine_libname}${CMAKE_SHARED_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")
        
      target_link_libraries(
        "bolt-omptarget.rtl.${tmachine_libname}"
        ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES} 
        ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
        dl
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")
    
      list(APPEND LIBBOLTTARGET_TESTED_PLUGINS
        "bolt-omptarget.rtl.${tmachine_libname}")

      # Report to the parent scope that we are building a plugin.
      set(LIBBOLTTARGET_SYSTEM_TARGETS 
        "${LIBBOLTTARGET_SYSTEM_TARGETS} ${tmachine_triple}" PARENT_SCOPE)
      set(LIBBOLTTARGET_TESTED_PLUGINS
        "${LIBBOLTTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
      
    else(LIBOMPTARGET_DEP_LIBFFI_FOUND)
      libomptarget_say("Not building ${tmachine_name} offloading plugin: libffi dependency not found.")
    endif(LIBOMPTARGET_DEP_LIBFFI_FOUND)
  else(LIBOMPTARGET_DEP_LIBELF_FOUND)
    libomptarget_say("Not building ${tmachine_name} offloading plugin: libelf dependency not found.")
  endif(LIBOMPTARGET_DEP_LIBELF_FOUND)
else()
  libomptarget_say("Not building ${tmachine_name} offloading plugin: machine not found in the system.")
endif()
endmacro()

add_subdirectory(aarch64)
add_subdirectory(cuda)
add_subdirectory(ppc64)
add_subdirectory(ppc64le)
add_subdirectory(ve)
add_subdirectory(x86_64)

# Make sure the parent scope can see the plugins that will be created.
set(LIBBOLTTARGET_SYSTEM_TARGETS "${LIBBOLTTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
set(LIBBOLTTARGET_TESTED_PLUGINS "${LIBBOLTTARGET_TESTED_PLUGINS}" PARENT_SCOPE)

