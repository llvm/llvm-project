add_flang_library(Fortran_main STATIC INSTALL_WITH_TOOLCHAIN
  Fortran_main.c
)
if (DEFINED MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  add_flang_library(Fortran_main.static STATIC INSTALL_WITH_TOOLCHAIN
    Fortran_main.c
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_flang_library(Fortran_main.dynamic STATIC INSTALL_WITH_TOOLCHAIN
    Fortran_main.c
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebug)
  add_flang_library(Fortran_main.static_dbg STATIC INSTALL_WITH_TOOLCHAIN
    Fortran_main.c
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
  add_flang_library(Fortran_main.dynamic_dbg STATIC INSTALL_WITH_TOOLCHAIN
    Fortran_main.c
  )
  add_dependencies(Fortran_main Fortran_main.static Fortran_main.dynamic
    Fortran_main.static_dbg Fortran_main.dynamic_dbg)
endif()
