// RUN: %clang %s -o %t -mmacosx-version-min=10.6 -framework CoreFoundation -DMAJOR=%macos_version_major -DMINOR=%macos_version_minor -DSUBMINOR=%macos_version_subminor
// RUN: %run %t

int __isOSVersionAtLeast(int Major, int Minor, int Subminor);

int main() {
  if (!__isOSVersionAtLeast(MAJOR, MINOR, SUBMINOR))
    return 1;
  if (__isOSVersionAtLeast(MAJOR, MINOR, SUBMINOR + 1))
    return 1;
  if (SUBMINOR && __isOSVersionAtLeast(MAJOR + 1, MINOR, SUBMINOR - 1))
    return 1;
  if (SUBMINOR && !__isOSVersionAtLeast(MAJOR, MINOR, SUBMINOR - 1))
    return 1;
  if (MAJOR && !__isOSVersionAtLeast(MAJOR - 1, MINOR + 1, SUBMINOR))
    return 1;

  return 0;
}
