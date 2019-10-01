#include <dpu.h>

int main() {
  struct dpu_rank_t *dpu_rank;

  if (dpu_alloc("backend=hw,cycleAccurate=true", &dpu_rank) != DPU_API_SUCCESS) {
    return -1;
  }

  struct dpu_t *dpu = dpu_get(dpu_rank, 0, 0);

  if (dpu_load_individual(dpu, DPU_EXE) != DPU_API_SUCCESS) {
    dpu_free(dpu_rank);
    return -2;
  }

  dpu_api_status_t status = dpu_boot_individual(dpu, SYNCHRONOUS);
  if (status != DPU_API_SUCCESS) {
    dpu_free(dpu_rank);
    return status;
  }

  dpu_free(dpu_rank);
  return 0;
}
