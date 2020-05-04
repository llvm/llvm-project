
#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void metadata_test1(void);

typedef struct test_meta_data_s {
  char *buf;
  amd_comgr_data_t data;
  amd_comgr_metadata_node_t root;
} test_meta_data_t;

int read_metadata(test_meta_data_t *meta_data, const char *file,
                  bool error_expected, bool display) {
  long size;
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t mkind = AMD_COMGR_METADATA_KIND_NULL;

  // Read input file
  char buffer[1024];
  snprintf(buffer, 1024, "%s/%s", TEST_OBJ_DIR, file);
  size = setBuf(buffer, &meta_data->buf);

  status =
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &meta_data->data);
  checkError(status, "amd_comgr_create_data");

  status = amd_comgr_set_data(meta_data->data, size, meta_data->buf);
  checkError(status, "amd_comgr_set_data");

  status = amd_comgr_set_data_name(meta_data->data, NULL);
  checkError(status, "amd_comgr_set_data_name");

  // Get metadata from data object
  if (display)
    printf("Get metadata from %s\n", file);

  status = amd_comgr_get_data_metadata(meta_data->data, &meta_data->root);
  if (error_expected)
    return 0;

  checkError(status, "amd_comgr_get_data_metadata");

  // the root must be map
  status = amd_comgr_get_metadata_kind(meta_data->root, &mkind);
  checkError(status, "amd_comgr_get_metadata_kind");
  if (mkind != AMD_COMGR_METADATA_KIND_MAP) {
    printf("Root is not map\n");
    exit(1);
  }

  if (display) {
    // print code object metadata
    int indent = 0;
    printf("Metadata for file %s : start\n", file);
    status = amd_comgr_iterate_map_metadata(meta_data->root, printEntry,
                                            (void *)&indent);
    checkError(status, "amd_comgr_iterate_map_metadata");
    printf("Metadata for file %s : end\n\n", file);
  }

  return 0;
}

int lookup_meta_data(test_meta_data_t *meta_data, const char *key,
                     amd_comgr_metadata_kind_t kind, void *data,
                     bool error_expected) {
  amd_comgr_status_t status;
  amd_comgr_metadata_node_t lookup_node;
  amd_comgr_metadata_kind_t lookup_kind;

  status = amd_comgr_metadata_lookup(meta_data->root, key, &lookup_node);
  checkError(status, "amd_comgr_metadata_lookup");

  status = amd_comgr_get_metadata_kind(lookup_node, &lookup_kind);
  if (error_expected)
    return 0;

  checkError(status, "amd_comgr_get_metadata_kind");
  if (lookup_kind != kind) {
    printf("Metadata kind mismatch in lookup\n");
    exit(1);
  }

  switch (kind) {
  case AMD_COMGR_METADATA_KIND_LIST: {
    size_t size = 0;
    size_t nentries = *((size_t *)data);

    status = amd_comgr_get_metadata_list_size(lookup_node, &size);
    checkError(status, "amd_comgr_get_metadata_list_size");
    if (size != nentries) {
      printf("List node size mismatch : expected %zu got %zu\n", nentries, size);
      exit(1);
    }
  } break;

  default:
    printf("Unknown kind\n");
    exit(1);
  }

  status = amd_comgr_destroy_metadata(lookup_node);
  checkError(status, "amd_comgr_destroy_metadata");

  return 0;
}

int close_meta_data(test_meta_data_t *meta_data) {
  amd_comgr_status_t status;

  status = amd_comgr_destroy_metadata(meta_data->root);
  checkError(status, "amd_comgr_destroy_metadata");

  status = amd_comgr_release_data(meta_data->data);
  checkError(status, "amd_comgr_release_data");
  free(meta_data->buf);

  memset(meta_data, 0, sizeof(test_meta_data_t));

  return 0;
}

int main(int argc, char *argv[]) {
  test_meta_data_t meta_data;

  memset(&meta_data, 0, sizeof(test_meta_data_t));

#define READ_METADATA(meta, file, is_error, display) do {   \
  read_metadata(&meta, file, is_error, display);            \
  close_meta_data(&meta);                                   \
}while(0)

#define LOOKUP_LIST_METADATA(meta, file, key, size, is_error) do {          \
  size_t n = size;                                                             \
  read_metadata(&meta, file, is_error, false);                              \
  lookup_meta_data(&meta, key, AMD_COMGR_METADATA_KIND_LIST, &n, is_error); \
  close_meta_data(&meta);                                                   \
}while(0)

  READ_METADATA(meta_data, "source1-v2.s.o", false, true);
  READ_METADATA(meta_data, "source2-v2.s.o", false, true);
  READ_METADATA(meta_data, "source1-v3.s.o", false, true);
  READ_METADATA(meta_data, "source2-v3.s.o", false, true);

  READ_METADATA(meta_data, "shared12-v2.so", true, true);

  LOOKUP_LIST_METADATA(meta_data, "shared12-v3.so", "amdhsa.printf", 1, false);
  LOOKUP_LIST_METADATA(meta_data, "shared12-v3.so", "amdhsa.kernels", 2, false);
  LOOKUP_LIST_METADATA(meta_data, "shared12-v3.so", "amdhsa.version", 2, false);

  LOOKUP_LIST_METADATA(meta_data, "shared14-v3.so", "amdhsa.version", 2, true);
  LOOKUP_LIST_METADATA(meta_data, "shared23-v3.so", "amdhsa.kernels", 2, true);

  printf("Metadata merge tests : passed\n");

  return 0;
}
