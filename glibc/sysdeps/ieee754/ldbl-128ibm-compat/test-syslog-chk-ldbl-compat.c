#define _FORTIFY_SOURCE 2
#define SYSLOG_FUNCTION __syslog_chk
#define SYSLOG_FUNCTION_PARAMS (LOG_DEBUG, 1, "%Lf\n", ld)
#define VSYSLOG_FUNCTION __vsyslog_chk
#define VSYSLOG_FUNCTION_PARAMS (LOG_DEBUG, 1, "%Lf\n", ap)
#include <test-syslog-ldbl-compat-template.c>
