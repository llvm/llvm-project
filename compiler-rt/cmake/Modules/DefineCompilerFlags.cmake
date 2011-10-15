# Define compiler flags

if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  #ADD_DEFINITIONS( -Wall -W -Werror -pedantic )
  ADD_DEFINITIONS( -std=c99 -Wall -Wextra -W -pedantic -Wno-unused-parameter )
endif( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
