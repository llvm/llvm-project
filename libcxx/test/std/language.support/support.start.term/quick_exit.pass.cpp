//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ::quick_exit and ::at_quick_exit were not implemented in older versions of macOS
// TODO: We should never be using `darwin` as the triple, but LLVM's config.guess script
//       guesses the host triple to be darwin instead of macosx when on macOS.
// XFAIL: target={{.+}}-apple-macosx10.{{13|14|15}}
// XFAIL: target={{.+}}-apple-macosx{{11|12|13|14}}{{(.+)?}}
// XFAIL: target={{.+}}-apple-darwin{{17|18|19|20|21|22|23}}{{(.+)?}}

// test quick_exit and at_quick_exit

#include <cstdlib>

void f() {}

int main(int, char**) {
    std::at_quick_exit(f);
    std::quick_exit(0);
    return 0;
}
