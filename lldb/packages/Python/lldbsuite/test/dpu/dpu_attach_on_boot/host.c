#include <dpu.h>

int main() {
  struct dpu_set_t set;
  struct dpu_set_t rank;
  struct dpu_set_t dpu;

  DPU_ASSERT(
      dpu_alloc(DPU_ALLOCATE_ALL, "backend=hw,cycleAccurate=true", &set));

  DPU_RANK_FOREACH(set, rank) { break; }

  DPU_FOREACH(set, dpu) { break; }

  DPU_ASSERT(dpu_load(rank, DPU_EXE, NULL));

  DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
  DPU_ASSERT(dpu_launch(rank, DPU_SYNCHRONOUS));
  DPU_ASSERT(dpu_launch(dpu, DPU_SYNCHRONOUS));
  DPU_ASSERT(dpu_launch(rank, DPU_SYNCHRONOUS));

  DPU_ASSERT(dpu_free(set));

  return 0;
}
