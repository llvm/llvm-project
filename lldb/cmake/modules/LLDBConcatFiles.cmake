# Concatenate the input files (positional args 2..N) into the output
# file (positional arg 1).
#
# Usage: cmake -P LLDBConcatFiles.cmake <output> <input1> [<input2> ...]

if(CMAKE_ARGC LESS 5)
  message(FATAL_ERROR
    "LLDBConcatFiles.cmake requires <output> and at least one <input>.")
endif()

set(_out "${CMAKE_ARGV3}")
file(WRITE "${_out}" "")

math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_i RANGE 4 ${_last})
  file(READ "${CMAKE_ARGV${_i}}" _data)
  file(APPEND "${_out}" "${_data}")
endforeach()
