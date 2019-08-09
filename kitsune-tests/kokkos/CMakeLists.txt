cmake_minimum_required(VERSION 3.4.3)

set(KOKKOS_LINK_LIBRARIES 
    "-lkokkos -ldl -lrt" 
    CACHE STRING "Required link libraries for kokkos.")

set(KOKKOS_INSTALL_PREFIX 
    "" 
    CACHE PATH "Install path prefix for kokkos.")

set(KOKKOS_IR_FLAGS ${CMAKE_CXX_FLAGS} -I${KOKKOS_INSTALL_PREFIX}/include -S -emit-llvm)

set(KITSUNE_IR_FLAGS -fkokkos ${CMAKE_CXX_FLAGS} -I${KOKKOS_INSTALL_PREFIX}/include -S -emit-llvm)

set(WC_FLAGS --lines --words)

file(GLOB SOURCES *.cpp)

foreach(cpp_file ${SOURCES})
  
  get_filename_component(target ${cpp_file} NAME_WE)

  # standard C++ target.  
  add_executable(${target}.kokkos ${cpp_file}) 
  target_include_directories(${target}.kokkos 
    PRIVATE ${KOKKOS_INSTALL_PREFIX}/include)
  target_link_libraries(${target}.kokkos 
    PRIVATE ${KOKKOS_LINK_LIBRARIES})

  add_custom_target(${target}.kokkos.ll ALL
    ${CMAKE_CXX_COMPILER} ${KOKKOS_IR_FLAGS} -o ${target}.kokkos.ll ${cpp_file}
    COMMAND /usr/bin/wc ${WC_FLAGS} ${target}.kokkos.ll > ${target}.kokkos.ll.stats
    BYPRODUCTS ${target}.kokkos.ll  ${target}.kokkos.ll.stats
    DEPENDS ${cpp_file})

  add_custom_target(${target}.kokkos.out 
    ALL
    COMMAND
    ${target}.kokkos > ${target}.kokkos.out
    BYPRODUCTS ${target}.kokkos.out 
    DEPENDS ${target.kokkos})

  add_test(NAME ${target}.kokkos.test 
    COMMAND ${target}.kokkos)


  # kitsune kokkos-aware target. 
  add_executable(${target}.kitsune ${cpp_file})
  target_include_directories(${target}.kitsune 
    PRIVATE ${KOKKOS_INSTALL_PREFIX}/include)
  target_compile_options(${target}.kitsune BEFORE
    PRIVATE "-fkokkos")
  target_link_libraries(${target}.kitsune
    PRIVATE ${KOKKOS_LINK_LIBRARIES})

  add_custom_target(${target}.kitsune.ll 
    ALL
    ${CMAKE_CXX_COMPILER} ${KITSUNE_IR_FLAGS} -o ${target}.kitsune.ll ${cpp_file}
    COMMAND /usr/bin/wc ${WC_FLAGS} ${target}.kitsune.ll > ${target}.kitsune.ll.stats
    BYPRODUCTS ${target}.kitsune.ll  ${target}.kitsune.ll.stats
    DEPENDS ${cpp_file})

  add_custom_target(${target}.kitsune.out 
    ALL
    COMMAND ${target}.kitsune > ${target}.kitsune.out
    BYPRODUCTS ${target}.kitsune.out 
    DEPENDS ${target.kitsune})

  add_test(NAME ${target}.kitsune.test 
    COMMAND ${target}.kitsune)

  add_test(${target}.compare-results  
    ${CMAKE_COMMAND} -E compare_files ${target}.kitsune.out ${target}.kokkos.out)

endforeach()

unset(WC_FLAGS)
