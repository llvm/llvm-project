// On platforms that have not implemented this yet, we provide an implementation
// that does not claim support for any features by leaving
// __riscv_feature_bits.feature_bits initialized to 0.

static void __init_riscv_feature_bits_ctor(void) CONSTRUCTOR_ATTRIBUTE;

// A constructor function that sets __riscv_feature_bits
// to the right values.  This needs to run only once.  This constructor is given
// the highest priority and it should run before constructors without the
// priority set.  However, it still runs after ifunc initializers and needs to
// be called explicitly there.

static void CONSTRUCTOR_ATTRIBUTE __init_riscv_feature_bits_ctor(void) {
  __init_riscv_feature_bits(0);
}

// PlatformArgs allows the platform to provide pre-computed data and access it
// without extra effort.
void __init_riscv_feature_bits(void *PlatformArgs) {}
