// On platforms that have not implemented this yet, we provide an implementation
// that does not claim support for any features by leaving
// __riscv_feature_bits initialized to 0.

static int initRISCVFeatureBits(unsigned long long features[],
                                void *PlatformArgs) {
  return -1;
}
