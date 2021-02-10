
#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void metadataTest1(void);

typedef struct test_meta_data_s {
  char *Buf;
  amd_comgr_data_t Data;
  amd_comgr_metadata_node_t Root;
} test_meta_data_t;

void read_metadata(test_meta_data_t *MetaData, const char *File,
                   bool ErrorExpected, bool Display) {
  long Size;
  amd_comgr_status_t Status;
  amd_comgr_metadata_kind_t Mkind = AMD_COMGR_METADATA_KIND_NULL;

  // Read input file
  char Buffer[1024];
  snprintf(Buffer, 1024, "%s/%s", TEST_OBJ_DIR, File);
  Size = setBuf(Buffer, &MetaData->Buf);

  Status =
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &MetaData->Data);
  checkError(Status, "amd_comgr_create_data");

  Status = amd_comgr_set_data(MetaData->Data, Size, MetaData->Buf);
  checkError(Status, "amd_comgr_set_data");

  Status = amd_comgr_set_data_name(MetaData->Data, NULL);
  checkError(Status, "amd_comgr_set_data_name");

  // Get metadata from data object
  if (Display) {
    printf("Get metadata from %s\n", File);
  }

  Status = amd_comgr_get_data_metadata(MetaData->Data, &MetaData->Root);
  if (!ErrorExpected && Status) {
    printf("Unexpected error from amd_comgr_get_data_metadata\n");
    exit(1);
  } else {
    return;
  }

  checkError(Status, "amd_comgr_get_data_metadata");

  // the root must be map
  Status = amd_comgr_get_metadata_kind(MetaData->Root, &Mkind);
  checkError(Status, "amd_comgr_get_metadata_kind");
  if (Mkind != AMD_COMGR_METADATA_KIND_MAP) {
    printf("Root is not map\n");
    exit(1);
  }

  if (Display) {
    // print code object metadata
    int Indent = 0;
    printf("Metadata for file %s : start\n", File);
    Status = amd_comgr_iterate_map_metadata(MetaData->Root, printEntry,
                                            (void *)&Indent);
    checkError(Status, "amd_comgr_iterate_map_metadata");
    printf("Metadata for file %s : end\n\n", File);
  }
}

void lookup_meta_data(test_meta_data_t *MetaData, const char *Key,
                      amd_comgr_metadata_kind_t Kind, void *Data,
                      bool ErrorExpected) {
  amd_comgr_status_t Status;
  amd_comgr_metadata_node_t LookupNode;
  amd_comgr_metadata_kind_t LookupKind;

  Status = amd_comgr_metadata_lookup(MetaData->Root, Key, &LookupNode);
  checkError(Status, "amd_comgr_metadata_lookup");

  Status = amd_comgr_get_metadata_kind(LookupNode, &LookupKind);
  if (!ErrorExpected && Status) {
    printf("Unexpected error from amd_comgr_get_metadata_kind\n");
    exit(1);
  } else {
    Status = amd_comgr_destroy_metadata(LookupNode);
    checkError(Status, "amd_comgr_destroy_metadata");
    return;
  }

  checkError(Status, "amd_comgr_get_metadata_kind");
  if (LookupKind != Kind) {
    printf("Metadata kind mismatch in lookup\n");
    exit(1);
  }

  switch (Kind) {
  case AMD_COMGR_METADATA_KIND_LIST: {
    size_t Size = 0;
    size_t Nentries = *((size_t *)Data);

    Status = amd_comgr_get_metadata_list_size(LookupNode, &Size);
    checkError(Status, "amd_comgr_get_metadata_list_size");
    if (Size != Nentries) {
      printf("List node size mismatch : expected %zu got %zu\n", Nentries,
             Size);
      exit(1);
    }
  } break;

  default:
    printf("Unknown kind\n");
    exit(1);
  }

  Status = amd_comgr_destroy_metadata(LookupNode);
  checkError(Status, "amd_comgr_destroy_metadata");
}

void close_meta_data(test_meta_data_t *MetaData) {
  amd_comgr_status_t Status;

  Status = amd_comgr_destroy_metadata(MetaData->Root);
  checkError(Status, "amd_comgr_destroy_metadata");

  Status = amd_comgr_release_data(MetaData->Data);
  checkError(Status, "amd_comgr_release_data");
  free(MetaData->Buf);

  memset(MetaData, 0, sizeof(test_meta_data_t));
}

int main(int argc, char *argv[]) {
  test_meta_data_t MetaData;

  memset(&MetaData, 0, sizeof(test_meta_data_t));

#define READ_METADATA(meta, file, is_error, display)                           \
  do {                                                                         \
    read_metadata(&meta, file, is_error, display);                             \
    close_meta_data(&meta);                                                    \
  } while (0)

#define LOOKUP_LIST_METADATA(meta, file, key, size, is_error)                  \
  do {                                                                         \
    size_t n = size;                                                           \
    read_metadata(&meta, file, is_error, false);                               \
    lookup_meta_data(&meta, key, AMD_COMGR_METADATA_KIND_LIST, &n, is_error);  \
    close_meta_data(&meta);                                                    \
  } while (0)

  READ_METADATA(MetaData, "source1-v2.o", false, true);
  READ_METADATA(MetaData, "source2-v2.o", false, true);
  READ_METADATA(MetaData, "source1-v3.o", false, true);
  READ_METADATA(MetaData, "source2-v3.o", false, true);

  READ_METADATA(MetaData, "shared12-v2.so", true, true);

  LOOKUP_LIST_METADATA(MetaData, "shared12-v3.so", "amdhsa.printf", 1, false);
  LOOKUP_LIST_METADATA(MetaData, "shared12-v3.so", "amdhsa.kernels", 2, false);
  LOOKUP_LIST_METADATA(MetaData, "shared12-v3.so", "amdhsa.version", 2, false);

  LOOKUP_LIST_METADATA(MetaData, "shared14-v3.so", "amdhsa.version", 2, true);
  LOOKUP_LIST_METADATA(MetaData, "shared23-v3.so", "amdhsa.kernels", 2, true);

  printf("Metadata merge tests : passed\n");

  return 0;
}
