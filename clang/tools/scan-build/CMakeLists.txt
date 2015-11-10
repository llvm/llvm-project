add_custom_target(scan-build ALL)

option(CLANG_INSTALL_SCANBUILD "Install the scan-build tool" ON)

if (WIN32 AND NOT CYGWIN)
  set(BinFiles
        scan-build.bat)
  set(LibexecFiles
        ccc-analyzer.bat
        c++-analyzer.bat)
else()
  set(BinFiles
        scan-build)
  set(LibexecFiles
        ccc-analyzer
        c++-analyzer)
  if (APPLE)
    set(BinFiles ${BinFiles}
         set-xcode-analyzer)
  endif()
endif()

set(ManPages
      scan-build.1)

set(ResourceFiles
      scanview.css
      sorttable.js)


if(CLANG_INSTALL_SCANBUILD)
  foreach(BinFile ${BinFiles})
    add_custom_command(TARGET scan-build PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/bin
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${BinFile}
                         ${CMAKE_BINARY_DIR}/bin/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${BinFile})
    install(PROGRAMS ${BinFile} DESTINATION bin)
  endforeach()

  foreach(LibexecFile ${LibexecFiles})
    add_custom_command(TARGET scan-build PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/libexec
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${LibexecFile}
                         ${CMAKE_BINARY_DIR}/libexec/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${LibexecFile})
    install(PROGRAMS ${LibexecFile} DESTINATION libexec)
  endforeach()

  foreach(ManPage ${ManPages})
    add_custom_command(TARGET scan-build PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/share/man/man1
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${ManPage}
                         ${CMAKE_BINARY_DIR}/share/man/man1/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ManPage})
    install(PROGRAMS ${ManPage} DESTINATION share/man/man1)
  endforeach()

  foreach(ResourceFile ${ResourceFiles})
    add_custom_command(TARGET scan-build PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/bin
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${ResourceFile}
                         ${CMAKE_BINARY_DIR}/bin/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ResourceFile})
    install(FILES ${ResourceFile} DESTINATION bin)
  endforeach()
endif()
