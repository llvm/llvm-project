//===- compile_hip_with_libcxx_test.c -------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test that embedded libc++ headers work with HIP compilation.
// This test verifies that HIPRTC-style compilation can use standard C++
// headers that don't require system C library headers.
//
// Supported headers: type_traits, limits, tuple, cstdint, initializer_list,
//                    concepts (C++20)
// NOT supported (require system C headers): optional, variant, ratio, array,
//                                           functional, cstring, cmath
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// HIP source that uses embedded libc++ headers (supported subset only)
const char *HipSource =
    "// Define HIP attributes since we use -nogpuinc\n"
    "#define __global__ __attribute__((global))\n"
    "#define __device__ __attribute__((device))\n"
    "\n"
    "// Supported headers (no system C library dependencies)\n"
    "#include <type_traits>\n"
    "#include <limits>\n"
    "#include <tuple>\n"
    "#include <cstdint>\n"
    "#include <initializer_list>\n"
    "\n"
    "// Compile-time tests using type_traits\n"
    "static_assert(std::is_integral<int>::value, \"int is integral\");\n"
    "static_assert(!std::is_integral<float>::value, \"float not integral\");\n"
    "static_assert(std::is_same<int, int>::value, \"int == int\");\n"
    "static_assert(std::is_pointer<int*>::value, \"int* is pointer\");\n"
    "\n"
    "// Compile-time tests using limits\n"
    "static_assert(std::numeric_limits<int>::is_integer, \"int is integer\");\n"
    "static_assert(std::numeric_limits<int>::max() > 0, \"int max > 0\");\n"
    "\n"
    "// Compile-time tests using tuple\n"
    "static_assert(std::tuple_size<std::tuple<int, float>>::value == 2,\n"
    "              \"tuple_size\");\n"
    "\n"
    "// Compile-time tests using cstdint\n"
    "static_assert(sizeof(std::int32_t) == 4, \"int32_t is 4 bytes\");\n"
    "static_assert(sizeof(std::int64_t) == 8, \"int64_t is 8 bytes\");\n"
    "\n"
    "// Template using enable_if\n"
    "template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>\n"
    "__device__ T square(T x) { return x * x; }\n"
    "\n"
    "// Template using conditional\n"
    "template<typename T>\n"
    "__device__ auto get_value() -> std::conditional_t<std::is_integral<T>::value, int, float> {\n"
    "    if constexpr (std::is_integral<T>::value) return 42;\n"
    "    else return 3.14f;\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void test_kernel(int *out) {\n"
    "    // Runtime tests\n"
    "    out[0] = std::is_same_v<int, int> ? 1 : 0;\n"
    "    out[1] = std::numeric_limits<int>::max() > 0 ? 1 : 0;\n"
    "    std::tuple<int, float> t{100, 3.14f};\n"
    "    out[2] = std::get<0>(t);\n"
    "    out[3] = square(7);  // 49\n"
    "    out[4] = get_value<int>();  // 42\n"
    "}\n";

int main(int Argc, char *Argv[]) {
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinkedBc, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  // Compile options: embedded libc++ headers are mapped to clang's default
  // include locations via VFS and injected as a fallback (-idirafter).
  // No explicit -I flags needed.
  const char *CompileOptions[] = {
      "-std=c++17",
      "-nogpuinc"                  // Don't use GPU-specific includes
  };
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  // Create source data
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, strlen(HipSource), HipSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "test_libcxx.hip");
  checkError(Status, "amd_comgr_set_data_name");

  // Create input data set
  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  // Create action info
  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(ActionInfo,
                                              "amdgcn-amd-amdhsa--gfx906");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_option_list(ActionInfo, CompileOptions,
                                                 CompileOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  // Compile to bitcode
  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, ActionInfo,
      DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action (compile to BC)");

  // Link bitcode
  Status = amd_comgr_create_data_set(&DataSetLinkedBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, ActionInfo,
                               DataSetBc, DataSetLinkedBc);
  checkError(Status, "amd_comgr_do_action (link BC)");

  // Generate relocatable
  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               ActionInfo, DataSetLinkedBc, DataSetReloc);
  checkError(Status, "amd_comgr_do_action (codegen to reloc)");

  // Link to executable
  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               ActionInfo, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action (link to exec)");

  printf("Successfully compiled HIP code with embedded libc++ headers\n");

  // Cleanup
  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinkedBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");

  return 0;
}
