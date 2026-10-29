set(LLVM_OPTIONAL_SOURCES
  CIROpDefinitions.cpp
  )

# Compile the TableGen-generated op definitions in shards to reduce the
# compile time of the largest CIR translation unit.
set(LLVM_TARGET_DEFINITIONS CIROpDefinitions.cpp)
math(EXPR CIR_LAST_SHARD_INDEX "${CLANG_CIR_OP_SHARD_COUNT} - 1")
foreach(index RANGE ${CIR_LAST_SHARD_INDEX})
  set(CIR_SHARDED_SRC CIROpDefinitions.${index}.cpp)
  list(APPEND CIR_SHARDED_SRCS ${CIR_SHARDED_SRC})
  tablegen(MLIR_SRC_SHARDER ${CIR_SHARDED_SRC}
    -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRCIROpsShardGen)

add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRDialectBytecode.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  CIROpenCLAttrs.cpp
  ${CIR_SHARDED_SRCS}

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIROpsShardGen
  MLIRCIRDialectBytecodeIncGen
  MLIRCIRTypeConstraintsIncGen
  MLIRCIREnumsGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRPtrMemorySpaceInterfaces
  clangAST
  )
