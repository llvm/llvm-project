//===-- Integration Test for Scudo ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

static const size_t ALLOC_SIZE = 128;

int main() {
  void *P = malloc(ALLOC_SIZE);
  if (P == nullptr) {
    return 1;
  }

  free(P);

  P = calloc(4, ALLOC_SIZE);
  if (P == nullptr) {
    return 2;
  }

  P = realloc(P, ALLOC_SIZE * 8);
  if (P == nullptr) {
    return 3;
  }

  free(P);

  P = aligned_alloc(64, ALLOC_SIZE);
  if (P == nullptr) {
    return 4;
  }

  free(P);

  return 0;
}
