# Construct LLVM version define
set( LLVM_VERSION_DEFINE "-DHAVE_LLVM=0x${LLVM_VERSION_MAJOR}0${LLVM_VERSION_MINOR}" )

# Setup prepare_builtins tools
set( LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  IRReader
  Support
)

if( LIBCLC_STANDALONE_BUILD )
  add_llvm_executable( prepare_builtins prepare-builtins.cpp )
  set( prepare_builtins_exe prepare_builtins PARENT_SCOPE )
  set( prepare_builtins_target prepare_builtins PARENT_SCOPE )
else()
  add_llvm_utility( prepare_builtins prepare-builtins.cpp )
  setup_host_tool( prepare_builtins PREPARE_BUILTINS prepare_builtins_exe prepare_builtins_target )
endif()

target_compile_definitions( prepare_builtins PRIVATE ${LLVM_VERSION_DEFINE} )
# These were not properly reported in early LLVM and we don't need them
target_compile_options( prepare_builtins PRIVATE -fno-rtti -fno-exceptions )
