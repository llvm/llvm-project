// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -std=c++20 -triple x86_64-apple-darwin %s 2>&1 | FileCheck %s

namespace GH24265 {
  enum class E_int { e };
  enum class E_long : long { e };

  void f() {
    E_int::e + E_long::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:5-[[@LINE]]:5}:"static_cast<int>("
                          // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:12-[[@LINE-1]]:12}:")"
                          // CHECK: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"static_cast<long>("
                          // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:24-[[@LINE-3]]:24}:")"
    E_int::e + 0; // CHECK: fix-it:{{.*}}:{[[@LINE]]:5-[[@LINE]]:5}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:12-[[@LINE-1]]:12}:")"

    0 * E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 / E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 % E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 + E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 - E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 << E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 >> E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 <=> E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:11-[[@LINE]]:11}:"static_cast<int>("
                    // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:18-[[@LINE-1]]:18}:")"
    0 < E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 > E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 <= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 >= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 == E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 != E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 & E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 ^ E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 | E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:9-[[@LINE]]:9}:"static_cast<int>("
                  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:16-[[@LINE-1]]:16}:")"
    0 && E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    0 || E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"

    int a;
    a *= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a /= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a %= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a += E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a -= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a <<= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:11-[[@LINE]]:11}:"static_cast<int>("
                    // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:18-[[@LINE-1]]:18}:")"
    a >>= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:11-[[@LINE]]:11}:"static_cast<int>("
                    // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:18-[[@LINE-1]]:18}:")"
    a &= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a ^= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"
    a |= E_int::e; // CHECK: fix-it:{{.*}}:{[[@LINE]]:10-[[@LINE]]:10}:"static_cast<int>("
                   // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:17-[[@LINE-1]]:17}:")"

    // TODO: These do not have the diagnostic yet
    E_int b;
    b *= 0;
    b /= 0;
    b %= 0;
    b += 0;
    b -= 0;
    b <<= 0;
    b >>= 0;
    b &= 0;
    b ^= 0;
    b |= 0;

    a = E_int::e;
    b = 0;

    E_int c = 0;
    int d = E_int::e;
  }
}
