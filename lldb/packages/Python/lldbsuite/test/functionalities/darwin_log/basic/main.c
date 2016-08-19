//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <os/log.h>
#include <stdio.h>

#include "../common/darwin_log_common.h"

int main(int argc, char** argv)
{
    os_log_t logger = os_log_create("org.llvm.lldb.test", "basic-test");
    if (!logger)
        return 1;

    // Note we cannot use the os_log() line as the breakpoint because, as of
    // the initial writing of this test, we get multiple breakpoints for that
    // line, which confuses the pexpect test logic.
    printf("About to log\n"); // break here
    os_log(logger, "Hello, world");

    // Sleep, as the darwin log reporting doesn't always happen until a bit
    // later.  We need the message to come out before the process terminates.
    sleep(FINAL_WAIT_SECONDS);

    return 0;
}
