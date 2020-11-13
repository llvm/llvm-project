if(LIBOMP_OMPT_SUPPORT)
  include_directories(${LIBOMP_INCLUDE_DIR})

  add_library(bolt-ompt-multiplex INTERFACE)
  target_include_directories(bolt-ompt-multiplex INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

  install(FILES ompt-multiplex.h DESTINATION include)

  add_subdirectory(tests)
endif()
