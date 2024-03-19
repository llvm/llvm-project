
#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct test_meta_data_s {
  char *Buf;
  amd_comgr_data_t Data;
  amd_comgr_metadata_node_t Root;
} test_meta_data_t;

void read_metadata(test_meta_data_t *MetaData, const char *File, bool IsErr) {
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
  printf("Get metadata from %s\n", File);

  Status = amd_comgr_get_data_metadata(MetaData->Data, &MetaData->Root);
  checkError(Status, "amd_comgr_get_data_metadata");

  // the root must be map
  Status = amd_comgr_get_metadata_kind(MetaData->Root, &Mkind);
  checkError(Status, "amd_comgr_get_metadata_kind");
  if (Mkind != AMD_COMGR_METADATA_KIND_MAP) {
    printf("Root is not map\n");
    exit(1);
  }

  // iterate code object metadata
  int Indent = 0;
  printf("Metadata for file %s : start\n", File);
  Status = amd_comgr_iterate_map_metadata(MetaData->Root, printEntry,
                                          (void *)&Indent);
  if (Status) {
    if (IsErr)
      return;
    checkError(Status, "amd_comgr_iterate_map_metadata");
  } else if (IsErr) {
    printf("Unexpected success from amd_comgr_iterate_map_metadata\n");
    exit(1);
  }
  printf("Metadata for file %s : end\n\n", File);
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

#define READ_METADATA(meta, file, is_error)                                    \
  do {                                                                         \
    read_metadata(&meta, file, is_error);                                      \
    close_meta_data(&meta);                                                    \
  } while (0)

  READ_METADATA(MetaData, "multiple-note-records.out", false);
  READ_METADATA(MetaData, "multiple-note-records-one-kernel.out", false);

  printf("Metadata Multiple MsgPacks tests : passed\n");
  return 0;
}
