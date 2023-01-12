include(FindGRPC)

function(generate_clang_protos_library LibraryName ProtoFile)
  # Take the first two args and forward the remaining to generate_proto_sources.
  cmake_parse_arguments(PARSE_ARGV 2 PROTO "" "" "")
  generate_proto_sources(ProtoSource ${ProtoFile} ${PROTO_UNPARSED_ARGUMENTS})
  set(LINKED_GRPC_LIBRARIES protobuf gpr grpc grpc++)

  if (ABSL_SYNCHRONIZATION_LIBRARY)
    list(APPEND LINKED_GRPC_LIBRARIES absl_synchronization)
  endif()
  add_clang_library(${LibraryName} ${ProtoSource}
    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC  ${LINKED_GRPC_LIBRARIES})

endfunction()
