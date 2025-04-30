#include "intr-rpc.h"
#define gsync_wait gsync_wait_intr
#define __gsync_wait __gsync_wait_intr
#include "RPC_gsync_wait.c"
