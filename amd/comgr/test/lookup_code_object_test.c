#include "amd_comgr.h"
#include "common.h"
#include <fcntl.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/types.h>

void buildFatBinary(const char *Input, const char *InputPath,
                    const char *OutputPath, const char *OptionsList[],
                    size_t OptionsCount) {
  char *Buffer;
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataFatBin;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Size = setBuf(InputPath, &Buffer);
  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, Size, Buffer);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, Input);
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");
  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_option_list(DataAction, OptionsList,
                                                 OptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");
  Status = amd_comgr_create_data_set(&DataFatBin);
  checkError(Status, "amd_comgr_create_data_set");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN,
                               DataAction, DataSetIn, DataFatBin);

#pragma GCC diagnostic pop

  checkError(Status, "amd_comgr_do_action");
  amd_comgr_data_t FatBinData;
  Status = amd_comgr_action_data_get_data(
      DataFatBin, AMD_COMGR_DATA_KIND_FATBIN, 0, &FatBinData);
  checkError(Status, "amd_comgr_action_data_get_data");
  dumpData(FatBinData, OutputPath);
  Status = amd_comgr_release_data(FatBinData);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataFatBin);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(Buffer);
}

void createFatBinary(const char *InputName, const char *InputPath,
                     const char *Output) {
  const char *Options1[] = {"--offload-arch=gfx701",
                            "--offload-arch=gfx702",
                            "--offload-arch=gfx801",
                            "--offload-arch=gfx802",
                            "--offload-arch=gfx803",
                            "--offload-arch=gfx810",
                            "--offload-arch=gfx900",
                            "--offload-arch=gfx902",
                            "--offload-arch=gfx904",
                            "--offload-arch=gfx906:sramecc+:xnack+",
                            "--offload-arch=gfx906:sramecc+:xnack-",
                            "--offload-arch=gfx906:sramecc-:xnack+",
                            "--offload-arch=gfx906:sramecc-:xnack-",
                            "--offload-arch=gfx906:xnack+",
                            "--offload-arch=gfx908:sramecc+:xnack+",
                            "--offload-arch=gfx908:sramecc+:xnack-",
                            "--offload-arch=gfx908:sramecc-:xnack+",
                            "--offload-arch=gfx908:sramecc-:xnack-",
                            "--offload-arch=gfx908:xnack-",
                            "-mcode-object-version=4"};
  size_t Options1Size = sizeof(Options1) / sizeof(Options1[0]);
  buildFatBinary("source1.hip", TEST_OBJ_DIR "/source1.hip",
                 TEST_OBJ_DIR "/source1.fatbin", Options1, Options1Size);
}

void fatbinTest(const char *FatBin, amd_comgr_data_kind_t Kind) {
  const char *IsaStrings[3] = {"amdgcn-amd-amdhsa--gfx700",
                               "amdgcn-amd-amdhsa--gfx904",
                               "amdgcn-amd-amdhsa--gfx908:xnack-"};

  amd_comgr_code_object_info_t QueryList1[1] = {{IsaStrings[0], 0, 0}};
  amd_comgr_code_object_info_t QueryList2[1] = {{IsaStrings[2], 0, 0}};
  amd_comgr_code_object_info_t QueryList3[3] = {
      {IsaStrings[0], 0, 0}, {IsaStrings[1], 0, 0}, {IsaStrings[2], 0, 0}};

#if defined(_WIN32) || defined(_WIN64)
  struct _stat st;
  _stat(FatBin, &st);
  int FD = _open(FatBin, _O_CREAT | _O_RDWR);
#else
  struct stat st;
  stat(FatBin, &st);
  int FD = open(FatBin, O_CREAT | O_RDWR, 0755);
#endif

  size_t size = st.st_size;
  if (FD < 0) {
    fail("open failed for %s with errno %d", FatBin, errno);
  }

  amd_comgr_data_t DataObject;
  amd_comgr_status_t Status;

  Status = amd_comgr_create_data(Kind, &DataObject);
  checkError(Status, "amd_comgr_create_data");

  Status = amd_comgr_set_data_from_file_slice(DataObject, FD, 0, size);
  checkError(Status, "amd_comgr_get_file_slice");

  Status = amd_comgr_lookup_code_object(DataObject, QueryList1, 1);
  checkError(Status, "amd_comgr_lookup_code_object");
  if (QueryList1->offset != 0 && QueryList1->size != 0) {
    fail("Lookup succeeded for non-existent code object");
  }

  Status = amd_comgr_lookup_code_object(DataObject, QueryList2, 1);
  checkError(Status, "amd_comgr_lookup_code_object");

  if (QueryList2->offset == 0 || QueryList2->size == 0) {
    fail("Lookup failed for existent code object");
  }

  Status = amd_comgr_lookup_code_object(DataObject, QueryList3, 3);
  checkError(Status, "amd_comgr_lookup_code_object");
  if (QueryList3[0].offset != 0 && QueryList3[0].size != 0) {
    fail("Lookup succeeded for non-existent code object");
  }

  if (QueryList3[1].offset == 0 || QueryList3[1].size == 0 ||
      QueryList3[2].offset == 0 || QueryList3[2].size == 0) {
    fail("Lookup failed for existent code object");
  }

#if defined(_WIN32) || defined(_WIN64)
  _close(FD);
#else
  close(FD);
#endif
}

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
#ifdef HIP_COMPILER
  createFatBinary("source1.hip", TEST_OBJ_DIR "/source1.hip",
                  TEST_OBJ_DIR "/source1.fatbin");

  fatbinTest(TEST_OBJ_DIR "/source1.fatbin", AMD_COMGR_DATA_KIND_FATBIN);
  fatbinTest(TEST_OBJ_DIR "/source1.fatbin", AMD_COMGR_DATA_KIND_BYTES);

  int Ret;
  if ((Ret = remove(TEST_OBJ_DIR "/source1.fatbin")) != 0) {
#if defined(_WIN32) || defined(_WIN64)
    if ((Ret = remove(TEST_OBJ_DIR "/source1.fatbin")) != 0) {
      fail("remove failed");
    }
#else
    fail("remove failed");
#endif
  }
#endif

  sharedObjectTest(AMD_COMGR_DATA_KIND_EXECUTABLE);
  sharedObjectTest(AMD_COMGR_DATA_KIND_BYTES);
}
