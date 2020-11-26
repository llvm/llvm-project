// RUN: clang-tidy %s -checks=-*,readability-else-after-return

// We aren't concerned about the output here, just want to ensure clang-tidy doesn't crash.
void foo() {
#if 1
  if (true) {
    return;
#else
  {
#endif
  } else {
    return;
  }

  if (true) {
#if 1
    return;
  } else {
#endif
    return;
  }
}
