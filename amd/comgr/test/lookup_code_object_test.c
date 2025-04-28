#include "amd_comgr.h"
#include "common.h"
#include <fcntl.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/types.h>

void sharedObjectTest(amd_comgr_data_kind_t Kind) {
  char *Buf;
  amd_comgr_data_t DataObject;
  amd_comgr_status_t Status;

  size_t Size = setBuf(TEST_OBJ_DIR "/shared.so", &Buf);
  Status = amd_comgr_create_data(Kind, &DataObject);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataObject, Size, Buf);
  checkError(Status, "amd_comgr_set_data");

  amd_comgr_code_object_info_t QueryList1[1] = {
      {"amdgcn-amd-amdhsa--gfx700", 0, 0}};

  amd_comgr_code_object_info_t QueryList2[1] = {
      {"amdgcn-amd-amdhsa--gfx900", 0, 0}};

  Status = amd_comgr_lookup_code_object(DataObject, QueryList1, 1);
  checkError(Status, "amd_comgr_lookup_code_object");

  if (QueryList1->offset != 0 || QueryList1->size != 0) {
    fail("Lookup succeeded for non-existent code object");
  }

  Status = amd_comgr_lookup_code_object(DataObject, QueryList2, 1);
  checkError(Status, "amd_comgr_lookup_code_object");

  if (QueryList2->offset != 0 || QueryList2->size == 0) {
    fail("Lookup failed for code object");
  }

  free(Buf);
}

int main(void) {
  sharedObjectTest(AMD_COMGR_DATA_KIND_EXECUTABLE);
  sharedObjectTest(AMD_COMGR_DATA_KIND_BYTES);
}
