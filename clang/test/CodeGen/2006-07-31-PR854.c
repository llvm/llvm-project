// RUN: %clang_cc1 -triple i686-linux-gnu -w %s -emit-llvm -o -

// PR854
  struct kernel_symbol {
    unsigned long value;
  };
  unsigned long loops_per_jiffy = (1<<12);
  static const char __kstrtab_loops_per_jiffy[]
__attribute__((section("__ksymtab_strings"))) = "loops_per_jiffy";
  static const struct kernel_symbol __ksymtab_loops_per_jiffy
__attribute__((__used__)) __attribute__((section("__ksymtab"))) = { (unsigned
long)&loops_per_jiffy, __kstrtab_loops_per_jiffy };
