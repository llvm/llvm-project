// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o /dev/null

// This test should not exhibit use-after-free in LoopInfo.

int a(void) {
  for (;;)
    for (;;)
      for (;;)
        for (;;)
          for (;;)
            for (;;)
              for (;;)
                for (;;)
                  for (;;)
                    ;
}
