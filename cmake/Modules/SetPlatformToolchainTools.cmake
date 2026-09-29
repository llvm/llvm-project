get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

if(CMAKE_SYSTEM_NAME MATCHES "AIX")
  foreach(lang IN LISTS languages)
    set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> -X32_64 qc <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_${lang}_ARCHIVE_APPEND "<CMAKE_AR> -X32_64 q <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_${lang}_ARCHIVE_FINISH "<CMAKE_RANLIB> -X32_64 <TARGET>")
  endforeach()
endif()
