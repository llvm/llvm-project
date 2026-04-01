add_aiir_library(AIIROpenACCMPCommon
AtomicInterfaces.cpp

ADDITIONAL_HEADER_DIRS
${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/OpenACCMPCommon/Interfaces

DEPENDS
AIIRAtomicInterfacesIncGen

LINK_LIBS PUBLIC
AIIRIR
)
