set(LLVM_NO_RTTI 1)

add_lldb_library(lldbCore
  Address.cpp
  AddressRange.cpp
  AddressResolver.cpp
  AddressResolverFileLine.cpp
  AddressResolverName.cpp
  ArchSpec.cpp
  Baton.cpp
  Broadcaster.cpp
  Communication.cpp
  Connection.cpp
  ConnectionFileDescriptor.cpp
  ConnectionMachPort.cpp
  ConnectionSharedMemory.cpp
  ConstString.cpp
  cxa_demangle.cpp  
  DataBufferHeap.cpp
  DataBufferMemoryMap.cpp
  DataEncoder.cpp
  DataExtractor.cpp
  Debugger.cpp
  Disassembler.cpp
  DynamicLoader.cpp
  EmulateInstruction.cpp
  Error.cpp
  Event.cpp
  FileLineResolver.cpp
  FileSpecList.cpp
  History.cpp
  InputReader.cpp
  InputReaderEZ.cpp
  InputReaderStack.cpp
  Language.cpp
  Listener.cpp
  Log.cpp
  Mangled.cpp
  Module.cpp
  ModuleChild.cpp
  ModuleList.cpp
  Opcode.cpp
  PluginManager.cpp
  RegisterValue.cpp
  RegularExpression.cpp
  Scalar.cpp
  SearchFilter.cpp
  Section.cpp
  SourceManager.cpp
  State.cpp
  Stream.cpp
  StreamAsynchronousIO.cpp
  StreamCallback.cpp
  StreamFile.cpp
  StreamString.cpp
  StringList.cpp
  Timer.cpp
  UserID.cpp
  UserSettingsController.cpp
  UUID.cpp
  Value.cpp
  ValueObject.cpp
  ValueObjectCast.cpp
  ValueObjectChild.cpp
  ValueObjectConstResult.cpp
  ValueObjectConstResultChild.cpp
  ValueObjectConstResultImpl.cpp
  ValueObjectDynamicValue.cpp
  ValueObjectList.cpp
  ValueObjectMemory.cpp
  ValueObjectRegister.cpp
  ValueObjectSyntheticFilter.cpp
  ValueObjectVariable.cpp
  VMRange.cpp
  #Windows.cpp
  )

#FIXME: Below we append -frtti to cxa_demangle.cpp (and let the compiler choose to ignore the
#       -fno-rtti that is added due to LLVM_NO_RTTI at the top of this file.) It would be
#       better to replace -fno-rtti with -frtti rather than just appending the latter option.
set_property(
  SOURCE cxa_demangle.cpp
  PROPERTY COMPILE_FLAGS -frtti
  )

