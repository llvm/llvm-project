int no_umbrella_A;

inline int has_warning(int x) {
  if (x > 0)
    return x;
  // Note: warning here is suppressed because this module is considered a
  // "system" module.
}
