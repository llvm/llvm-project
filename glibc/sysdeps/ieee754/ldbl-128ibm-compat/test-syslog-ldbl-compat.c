#define _FORTIFY_SOURCE 0
#define SYSLOG_FUNCTION syslog
#define SYSLOG_FUNCTION_PARAMS (LOG_DEBUG, "%Lf\n", ld)
#define VSYSLOG_FUNCTION vsyslog
#define VSYSLOG_FUNCTION_PARAMS (LOG_DEBUG, "%Lf\n", ap)
#include <test-syslog-ldbl-compat-template.c>
