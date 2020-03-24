project(KitsuneTests)

enable_testing()

cmake_minimum_required(VERSION 3.4.3)

set(KITSUNE_ENABLE_FORALL_TESTS OFF CACHE BOOL "Build the 'forall'-centric regressions")
set(KITSUNE_ENABLE_KOKKOS_TESTS ON CACHE  BOOL "Build the kokkos-centric regressions")
set(KITSUNE_ENABLE_FLECSI_TESTS OFF CACHE BOOL "Build the flecsi-centric regressions")

set(KITSUNE_INSTALL_PREFIX "" CACHE PATH "Kitsune toolchain installation path prefix")

if (KITSUNE_ENABLE_FORALL_TESTS)
   add_subdirectory(forall)
endif()

if (KITSUNE_ENABLE_KOKKOS_TESTS)
   add_subdirectory(kokkos)
endif()

if (KITSUNE_ENABLE_FLECSI_TESTS)
   add_subdirectory(flecsi)
endif()
