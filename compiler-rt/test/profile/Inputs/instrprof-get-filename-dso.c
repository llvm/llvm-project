const char *__llvm_profile_get_filename(void);

const char *get_filename_from_DSO(void) {
  return __llvm_profile_get_filename();
}
