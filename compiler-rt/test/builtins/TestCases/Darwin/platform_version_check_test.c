// RUN: %clang %s -o %t -mmacosx-version-min=10.7 -framework CoreFoundation -DMAJOR=%macos_version_major -DMINOR=%macos_version_minor -DSUBMINOR=%macos_version_subminor
// RUN: %run %t

typedef int int32_t;
typedef unsigned int uint32_t;

int32_t __isPlatformVersionAtLeast(uint32_t Platform, uint32_t Major,
                                   uint32_t Minor, uint32_t Subminor);

int32_t __isPlatformOrVariantPlatformVersionAtLeast(
    uint32_t Platform, uint32_t Major, uint32_t Minor, uint32_t Subminor,
    uint32_t Platform2, uint32_t Major2, uint32_t Minor2, uint32_t Subminor2);

void exit(int status);

#define PLATFORM_MACOS 1
#define PLATFORM_IOS 2

int32_t check(uint32_t Major, uint32_t Minor, uint32_t Subminor) {
  int32_t Result =
      __isPlatformVersionAtLeast(PLATFORM_MACOS, Major, Minor, Subminor);
  int32_t ResultVariant = __isPlatformOrVariantPlatformVersionAtLeast(
      PLATFORM_MACOS, Major, Minor, Subminor, PLATFORM_IOS, 13, 0, 0);
  if (Result != ResultVariant)
    exit(-1);
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
