#ifndef FEATURE1_H
#define FEATURE1_H
#include <feature-availability.h>

static struct __AvailabilityDomain feature1 __attribute__((availability_domain(feature1))) = {__AVAILABILITY_DOMAIN_ENABLED, 0};

#endif
