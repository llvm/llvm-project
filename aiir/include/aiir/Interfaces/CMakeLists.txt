add_aiir_interface(AlignmentAttrInterface)
add_aiir_interface(CallInterfaces)
add_aiir_interface(CastInterfaces)
add_aiir_interface(ControlFlowInterfaces)
add_aiir_interface(DerivedAttributeOpInterface)
add_aiir_interface(DestinationStyleOpInterface)
add_aiir_interface(FunctionInterfaces)
add_aiir_interface(IndexingMapOpInterface)
add_aiir_interface(InferIntRangeInterface)
add_aiir_interface(InferStridedMetadataInterface)
add_aiir_interface(InferTypeOpInterface)
add_aiir_interface(LoopLikeInterface)
add_aiir_interface(MemOpInterfaces)
add_aiir_interface(ParallelCombiningOpInterface)
add_aiir_interface(RuntimeVerifiableOpInterface)
add_aiir_interface(ShapedOpInterfaces)
add_aiir_interface(SideEffectInterfaces)
add_aiir_interface(SubsetOpInterface)
add_aiir_interface(TilingInterface)
add_aiir_interface(ValueBoundsOpInterface)
add_aiir_interface(VectorInterfaces)
add_aiir_interface(ViewLikeInterface)

set(LLVM_TARGET_DEFINITIONS MemorySlotInterfaces.td)
aiir_tablegen(MemorySlotOpInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(MemorySlotOpInterfaces.cpp.inc -gen-op-interface-defs)
aiir_tablegen(MemorySlotTypeInterfaces.h.inc -gen-type-interface-decls)
aiir_tablegen(MemorySlotTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_aiir_generic_tablegen_target(AIIRMemorySlotInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS DataLayoutInterfaces.td)
aiir_tablegen(DataLayoutAttrInterface.h.inc -gen-attr-interface-decls)
aiir_tablegen(DataLayoutAttrInterface.cpp.inc -gen-attr-interface-defs)
aiir_tablegen(DataLayoutOpInterface.h.inc -gen-op-interface-decls)
aiir_tablegen(DataLayoutOpInterface.cpp.inc -gen-op-interface-defs)
aiir_tablegen(DataLayoutTypeInterface.h.inc -gen-type-interface-decls)
aiir_tablegen(DataLayoutTypeInterface.cpp.inc -gen-type-interface-defs)
aiir_tablegen(DataLayoutDialectInterface.h.inc -gen-dialect-interface-decls)
add_aiir_generic_tablegen_target(AIIRDataLayoutInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS DialectFoldInterface.td)
aiir_tablegen(DialectFoldInterface.h.inc -gen-dialect-interface-decls)
add_aiir_generic_tablegen_target(AIIRDialectFoldInterfaceIncGen)


add_aiir_doc(DataLayoutInterfaces
  DataLayoutAttrInterface
  Interfaces/
  -gen-attr-interface-docs)

add_aiir_doc(DataLayoutInterfaces
  DataLayoutTypeInterface
  Interfaces/
  -gen-type-interface-docs)

add_aiir_doc(DataLayoutInterfaces
  DataLayoutOpInterface
  Interfaces/
  -gen-op-interface-docs)
