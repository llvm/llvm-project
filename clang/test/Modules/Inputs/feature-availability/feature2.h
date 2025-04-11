#ifndef FEATURE2_H
#define FEATURE2_H
#include <feature-availability.h>
#include "feature1.h"

static struct __AvailabilityDomain feature2 __attribute__((availability_domain(feature2))) = {__AVAILABILITY_DOMAIN_DISABLED, 0};

#endif
