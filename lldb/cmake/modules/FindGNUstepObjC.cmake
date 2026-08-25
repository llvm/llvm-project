#.rst:
# FindGNUstepObjC
# ---------------
#
# Find the GNUstep libobjc2 shared library.

set(gnustep_install_dir "")

if (UNIX)
  set(gnustep_lib lib/libobjc.so)
  set(gnustep_header include/objc/runtime.h)
  if (GNUstepObjC_DIR)
    if (EXISTS "${GNUstepObjC_DIR}/${gnustep_lib}" AND
        EXISTS "${GNUstepObjC_DIR}/${gnustep_header}")
      set(gnustep_install_dir ${GNUstepObjC_DIR})
    endif()
  else()
    set(gnustep_install_dir)
    find_path(gnustep_install_dir NAMES lib/libobjc.so include/objc/runtime.h)
  endif()
  if (gnustep_install_dir)
    set(GNUstepObjC_FOUND TRUE)
  endif()
elseif (WIN32)
  # MSVC libobjc2 installs lib/objc.dll; MinGW uses GNU naming (libobjc.dll,
  # possibly versioned; MSYS2 ships libobjc-4.6.dll) and usually puts it in
  # bin/.
  set(gnustep_header include/objc/runtime.h)
  if (GNUstepObjC_DIR)
    set(gnustep_install_dir ${GNUstepObjC_DIR})
  else()
    set(gnustep_install_dir "C:/Program Files (x86)/libobjc")
  endif()
  if (EXISTS "${gnustep_install_dir}/${gnustep_header}")
    # Prefer lib/, where the DLL sits beside its import library. One pattern
    # at a time: file(GLOB) sorts the union of its patterns, so passing them
    # together would not preserve this order.
    foreach (gnustep_lib_dir lib bin)
      foreach (gnustep_lib_name objc.dll libobjc.dll libobjc-[0-9]*.dll)
        file(GLOB gnustep_lib_matches
          RELATIVE "${gnustep_install_dir}"
          "${gnustep_install_dir}/${gnustep_lib_dir}/${gnustep_lib_name}")
        if (gnustep_lib_matches)
          list(GET gnustep_lib_matches 0 gnustep_lib)
          set(GNUstepObjC_FOUND TRUE)
          break()
        endif()
      endforeach()
      if (GNUstepObjC_FOUND)
        break()
      endif()
    endforeach()
  endif()
endif()

if (GNUstepObjC_FOUND)
  set(GNUstepObjC_DIR ${gnustep_install_dir})
  message(STATUS "Found GNUstep ObjC runtime: ${GNUstepObjC_DIR}/${gnustep_lib}")
endif()
