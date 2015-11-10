add_custom_target(scan-view ALL)

option(CLANG_INSTALL_SCANVIEW "Install the scan-view tool" ON)

set(BinFiles
      Reporter.py
      ScanView.py
      scan-view
      startfile.py)

set(ResourceFiles
      Resources/FileRadar.scpt
      Resources/GetRadarVersion.scpt
      Resources/bugcatcher.ico)

if(CLANG_INSTALL_SCANVIEW)
  foreach(BinFile ${BinFiles})
    add_custom_command(TARGET scan-view PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/bin
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${BinFile}
                         ${CMAKE_BINARY_DIR}/bin/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${BinFile})
    install(PROGRAMS ${BinFile} DESTINATION bin)
  endforeach()

  foreach(ResourceFile ${ResourceFiles})
    add_custom_command(TARGET scan-view PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/share/scan-view
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${ResourceFile}
                         ${CMAKE_BINARY_DIR}/share/scan-view/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ResourceFile})
    install(FILES ${ResourceFile} DESTINATION share/scan-view)
  endforeach()
endif()
