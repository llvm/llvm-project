void __init_cpu_features_resolver(unsigned long hwcap,
                                  const __ifunc_arg_t *arg) {
  if (__aarch64_cpu_features.features)
    return;
  __init_cpu_features_constructor(hwcap, arg);
}

void CONSTRUCTOR_ATTRIBUTE __init_cpu_features(void) {
  // CPU features already initialized.
  if (__aarch64_cpu_features.features)
    return;
  __ifunc_arg_t arg;
  arg._size = sizeof(__ifunc_arg_t);
  arg._hwcap = getauxval(AT_HWCAP);
  arg._hwcap2 = getauxval(AT_HWCAP2);
  __init_cpu_features_constructor(hwcap | _IFUNC_ARG_HWCAP, &arg);
}
