// For baremetal platforms, we don't really initialise '__aarch64_cpu_features',
// with exception of FEAT_SME that we can get from '__arm_sme_state'.

#define SME_STATE_X0_HAS_SME (1ULL << 63)

typedef unsigned long long u64;
_Static_assert(sizeof(u64) == 8, "u64 should be 64-bits");

// The libc/runtime for baremetal targets should provide an implementation of
// __arm_sme_state. LLVM's compiler-rt does not provide this function for
// baremetal as checking CPU features is implementation defined.
//
// See:
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#811__arm_sme_state
extern struct {
  u64 x0;
  u64 x1;
} __arm_sme_state(void);

void __init_cpu_features_resolver(unsigned long hwcap,
                                  const __ifunc_arg_t *arg) {}

void CONSTRUCTOR_ATTRIBUTE __init_cpu_features(void) {
  // CPU features already initialized.
  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;

  _Bool has_sme = (__arm_sme_state().x0 & SME_STATE_X0_HAS_SME) != 0;

  u64 feat = 0;
  if (has_sme)
    feat |= 1ULL << FEAT_SME;

  __atomic_store_n(&__aarch64_cpu_features.features, feat, __ATOMIC_RELAXED);
}
