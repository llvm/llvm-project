#include <dpu.h>

int main() {
  struct dpu_set_t set;
  struct dpu_set_t dpu;

  if (dpu_alloc(1, "backend=hw,cycleAccurate=true", &set) != DPU_API_SUCCESS) {
    DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set));
  }

  DPU_ASSERT(dpu_load(dpu, DPU_EXE, NULL));
  DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));

  DPU_ASSERT(dpu_free(set));

  return 0;
}
