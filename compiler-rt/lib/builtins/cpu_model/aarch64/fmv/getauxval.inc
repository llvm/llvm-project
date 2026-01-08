void __init_cpu_features_resolver(unsigned long hwcap,
                                  const __ifunc_arg_t *arg) {
  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;
  __init_cpu_features_constructor(hwcap, arg);
}

void CONSTRUCTOR_ATTRIBUTE __init_cpu_features(void) {
  // CPU features already initialized.
  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;

  unsigned long hwcap = getauxval(AT_HWCAP);
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  unsigned long hwcap3 = getauxval(AT_HWCAP3);
  unsigned long hwcap4 = getauxval(AT_HWCAP4);

  __ifunc_arg_t arg;
  arg._size = sizeof(__ifunc_arg_t);
  arg._hwcap = hwcap;
  arg._hwcap2 = hwcap2;
  arg._hwcap3 = hwcap3;
  arg._hwcap4 = hwcap4;
  __init_cpu_features_constructor(hwcap | _IFUNC_ARG_HWCAP, &arg);
}
