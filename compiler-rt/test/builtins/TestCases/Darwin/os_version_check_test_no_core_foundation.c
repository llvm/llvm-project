// RUN: %clang %s -o %t -mmacosx-version-min=10.6
// RUN: %run %t

int __isOSVersionAtLeast(int Major, int Minor, int Subminor);

int main() {
  // When CoreFoundation isn't linked, we expect the system version to be 0, 0,
  // 0.
  if (__isOSVersionAtLeast(1, 0, 0))
    return 1;
  return 0;
}
