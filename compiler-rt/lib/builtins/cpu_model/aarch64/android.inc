#include <string.h>
#include <sys/system_properties.h>

static bool __isExynos9810(void) {
  char arch[PROP_VALUE_MAX];
  return __system_property_get("ro.arch", arch) > 0 &&
         strncmp(arch, "exynos9810", sizeof("exynos9810") - 1) == 0;
}
