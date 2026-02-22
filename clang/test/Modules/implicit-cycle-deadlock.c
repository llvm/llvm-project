// This test checks that implicit modules do not encounter a deadlock on a dependency cycle.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %python %t/run_concurrently.py \
// RUN:   "not %clang -fsyntax-only -fmodules -fmodules-cache-path=%t/cache -fimplicit-modules-lock-timeout=3 %t/tu1.c 2> %t/err1" \
// RUN:   "not %clang -fsyntax-only -fmodules -fmodules-cache-path=%t/cache -fimplicit-modules-lock-timeout=3 %t/tu2.c 2> %t/err2"

// RUN: FileCheck %s --input-file %t/err1 --check-prefix=CHECK1
// RUN: FileCheck %s --input-file %t/err2 --check-prefix=CHECK2
// CHECK1: fatal error: cyclic dependency in module 'M': M -> N -> M
// CHECK2: fatal error: cyclic dependency in module 'N': N -> M -> N

//--- run_concurrently.py
import subprocess, sys, threading

def run(cmd):
    subprocess.run(cmd, shell=True)

threads = [threading.Thread(target=run, args=(cmd,)) for cmd in sys.argv]
for t in threads: t.start()
for t in threads: t.join()
//--- tu1.c
#include "m.h"
//--- tu2.c
#include "n.h"
//--- module.modulemap
module M { header "m.h" }
module N { header "n.h" }
//--- m.h
#pragma clang __debug sleep // Give enough time for tu2.c to start building N.
#include "n.h"
//--- n.h
#pragma clang __debug sleep // Give enough time for tu1.c to start building M.
#include "m.h"
