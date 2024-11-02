// RUN: %clang %s -o %t -mmacosx-version-min=10.7 -framework CoreFoundation -DMAJOR=%macos_version_major -DMINOR=%macos_version_minor -DSUBMINOR=%macos_version_subminor
// RUN: %run %t

typedef int int32_t;
typedef unsigned int uint32_t;

int32_t __isPlatformVersionAtLeast(uint32_t Platform, uint32_t Major,
                                   uint32_t Minor, uint32_t Subminor);

#define PLATFORM_MACOS 1

int32_t check(uint32_t Major, uint32_t Minor, uint32_t Subminor) {
  int32_t Result =
      __isPlatformVersionAtLeast(PLATFORM_MACOS, Major, Minor, Subminor);
  return Result;
}

int main() {
  if (!check(MAJOR, MINOR, SUBMINOR))
    return 1;
  if (check(MAJOR, MINOR, SUBMINOR + 1))
    return 1;
  if (SUBMINOR && check(MAJOR + 1, MINOR, SUBMINOR - 1))
    return 1;
  if (SUBMINOR && !check(MAJOR, MINOR, SUBMINOR - 1))
    return 1;
  if (MAJOR && !check(MAJOR - 1, MINOR + 1, SUBMINOR))
    return 1;

  return 0;
}
