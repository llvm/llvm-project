//===-- Main entry into the loader interface ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file opens a device image passed on the command line and passes it to
// one of the loader implementations for launch.
//
//===----------------------------------------------------------------------===//

#include "Loader.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("USAGE: ./loader <device_image> <args>, ...\n");
    return EXIT_SUCCESS;
  }

  // TODO: We should perform some validation on the file.
  FILE *file = fopen(argv[1], "r");

  if (!file) {
    fprintf(stderr, "Failed to open image file %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  fseek(file, 0, SEEK_END);
  const auto size = ftell(file);
  fseek(file, 0, SEEK_SET);

  void *image = malloc(size * sizeof(char));
  fread(image, sizeof(char), size, file);
  fclose(file);

  // Drop the loader from the program arguments.
  int ret = load(argc - 1, &argv[1], image, size);

  free(image);
  return ret;
}
