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
#include <string>
#include <vector>

int main(int argc, char **argv, char **envp) {
  if (argc < 2) {
    printf("USAGE: ./loader [--threads <n>, --blocks <n>] <device_image> "
           "<args>, ...\n");
    return EXIT_SUCCESS;
  }

  int offset = 0;
  FILE *file = nullptr;
  char *ptr;
  LaunchParameters params = {1, 1, 1, 1, 1, 1};
  while (!file && ++offset < argc) {
    if (argv[offset] == std::string("--threads") ||
        argv[offset] == std::string("--threads-x")) {
      params.num_threads_x =
          offset + 1 < argc ? strtoul(argv[offset + 1], &ptr, 10) : 1;
      offset++;
      continue;
    } else if (argv[offset] == std::string("--threads-y")) {
      params.num_threads_y =
          offset + 1 < argc ? strtoul(argv[offset + 1], &ptr, 10) : 1;
      offset++;
      continue;
    } else if (argv[offset] == std::string("--threads-z")) {
      params.num_threads_z =
          offset + 1 < argc ? strtoul(argv[offset + 1], &ptr, 10) : 1;
      offset++;
      continue;
    } else if (argv[offset] == std::string("--blocks") ||
               argv[offset] == std::string("--blocks-x")) {
      params.num_blocks_x =
          offset + 1 < argc ? strtoul(argv[offset + 1], &ptr, 10) : 1;
      offset++;
      continue;
    } else if (argv[offset] == std::string("--blocks-y")) {
      params.num_blocks_y =
          offset + 1 < argc ? strtoul(argv[offset + 1], &ptr, 10) : 1;
      offset++;
      continue;
    } else if (argv[offset] == std::string("--blocks-z")) {
      params.num_blocks_z =
          offset + 1 < argc ? strtoul(argv[offset + 1], &ptr, 10) : 1;
      offset++;
      continue;
    } else {
      file = fopen(argv[offset], "r");
      if (!file) {
        fprintf(stderr, "Failed to open image file '%s'\n", argv[offset]);
        return EXIT_FAILURE;
      }
      break;
    }
  }

  if (!file) {
    fprintf(stderr, "No image file provided\n");
    return EXIT_FAILURE;
  }

  // TODO: We should perform some validation on the file.
  fseek(file, 0, SEEK_END);
  const auto size = ftell(file);
  fseek(file, 0, SEEK_SET);

  void *image = malloc(size * sizeof(char));
  fread(image, sizeof(char), size, file);
  fclose(file);

  // Drop the loader from the program arguments.
  int ret = load(argc - offset, &argv[offset], envp, image, size, params);

  free(image);
  return ret;
}
