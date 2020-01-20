#include <dpu.h>

int main() {
  struct dpu_set_t set;
  struct dpu_set_t dpu;

  DPU_ASSERT(dpu_alloc(1, "backend=hw,cycleAccurate=true", &set));
  DPU_ASSERT(dpu_load(set, DPU_EXE, NULL));

  DPU_FOREACH(set, dpu) { DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS)); }
  DPU_ASSERT(dpu_free(set));

  return 0;
}
