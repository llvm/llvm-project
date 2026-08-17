#include <stdint.h>

// The libc/runtime for baremetal targets should provide an implementation of
// __arm_sme_state. LLVM's compiler-rt does not provide this function for
// baremetal as checking CPU features is implementation defined.
//
// See:
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#811__arm_sme_state
extern struct {
  uint64_t x0;
  uint64_t x1;
} __arm_sme_state(void);

// __arm_sme_state:
// Bit 63 of X0 is set to one iff the current thread has access to SME.
#define SME_STATE_HAS_SME_BIT_X0 (1ULL << 63)

void __init_cpu_features_resolver(unsigned long hwcap,
                                  const __ifunc_arg_t *arg) {}

void CONSTRUCTOR_ATTRIBUTE __init_cpu_features(void) {
  // CPU features already initialized.
  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;

  // For baremetal platforms, we don't really initialise
  // '__aarch64_cpu_features', with exception of FEAT_SME that we can get from
  // '__arm_sme_state'.
  _Bool has_sme = (__arm_sme_state().x0 & SME_STATE_HAS_SME_BIT_X0) != 0;

  uint64_t feat = 0;
  if (has_sme)
    feat |= 1ULL << FEAT_SME;

  __atomic_store_n(&__aarch64_cpu_features.features, feat, __ATOMIC_RELAXED);
}
