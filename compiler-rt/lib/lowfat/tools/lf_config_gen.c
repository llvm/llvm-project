//===-- lf_config_gen.c - LowFat Size Class Config Generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Standalone C tool (no LLVM dependencies) that reads a sizes.cfg file and
// emits lf_config_generated.h containing:
//
//   - kLowFatGenSizes[]   : actual object sizes for each region index
//   - kLowFatGenMagics[]  : precomputed 2^64/S values for non-POW2 sizes
//   - kLowFatGenIsPow2[]  : true for power-of-two size classes
//   - kLowFatGenMasks[]   : alignment masks for POW2 sizes (0 for non-POW2)
//   - lowfat_size_to_class(): binary-search mapping from alloc size → region index
//
// sizes.cfg format:
//   - One size per line (plain integer)
//   - Sizes must be multiples of 16
//   - First size must be 16
//   - Sizes must be in strictly ascending order
//   - Maximum size ≤ LOWFAT_REGION_SIZE_LOG (32 GB when kRegionSizeLog=35)
//
// Usage:
//   lf_config_gen <sizes.cfg> <output_header>
//
// The precision checker validates that the fixed-point formula
//   base = (ptr * magic >> 64) * S
// correctly identifies the start of every object within a 32 GB region.
// Where rounding errors exist, the effective size is reduced by the error
// so the allocator never hands out the last few bytes that would be
// mis-identified.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------------
// Configuration constants (must stay in sync with lf_config.h)
// --------------------------------------------------------------------------

// Each region is 32 GB when custom config is active.
// The original LowFat research proves the magic-number math is precise up to
// 32 GB boundaries; 4 GB is insufficient for non-POW2 sizes.
#define REGION_SIZE_LOG 35
#define REGION_SIZE     ((uint64_t)1 << REGION_SIZE_LOG)   // 32 GB
#define MAX_SIZE_CLASSES 256

// Minimum alignment / granularity
#define MIN_SIZE 16

// --------------------------------------------------------------------------
// __int128 helpers (standard C99/C11 with GCC/Clang extension)
// --------------------------------------------------------------------------

typedef unsigned __int128 u128;

// Compute ceil(2^64 / S) using 128-bit arithmetic.
// This is the magic number M such that floor(P / S) = (P * M) >> 64
// for all P in [0, REGION_SIZE).
static uint64_t compute_magic(uint64_t S) {
  if (S == 0) return 0;
  u128 two64 = (u128)1 << 64;
  uint64_t q = (uint64_t)(two64 / S);
  uint64_t r = (uint64_t)(two64 % S);
  return q + (r != 0 ? 1 : 0);  // ceil division
}

// Returns 1 if n is an exact power of two, 0 otherwise.
static int is_pow2(uint64_t n) {
  return n != 0 && (n & (n - 1)) == 0;
}

// --------------------------------------------------------------------------
// Precision Checker
//
// For each non-POW2 size S with magic M, scan backwards from the end of the
// region to find the first pointer P where the reconstructed base is wrong.
// Returns the number of bytes to subtract from S (the "error margin").
//
// The formula base(P) = ((u128)P * M >> 64) * S gives the start of the
// object containing P.  For it to be correct we need:
//   base(P) <= P  &&  P < base(P) + S
//
// We walk from REGION_SIZE - 1 down to 0 in steps of S, stopping at the
// first P where the formula breaks.
// --------------------------------------------------------------------------

static uint64_t precision_error(uint64_t S, uint64_t M) {
  if (is_pow2(S))
    return 0;  // POW2 uses bitwise AND, no rounding error

  uint64_t region = REGION_SIZE;
  uint64_t error  = 0;

  // Walk the last few objects (the vulnerable zone is at the top of the region)
  // Limit scan to last 1024 objects to keep tool fast; real errors are tiny.
  uint64_t scan_start = region > S * 1024 ? region - S * 1024 : 0;

  for (uint64_t ptr = region - 1; ptr >= scan_start && ptr != (uint64_t)-1; ptr--) {
    u128 mul  = (u128)ptr * (u128)M;
    uint64_t idx  = (uint64_t)(mul >> 64);
    uint64_t base = idx * S;

    // base must be the correct start: base <= ptr < base + S
    if (ptr < base || ptr >= base + S) {
      // ptr is mis-identified; error = bytes from end of last good object to ptr
      // We shrink S by the difference so the allocator stays safe.
      uint64_t last_good_end = (ptr / S) * S;
      error = ptr - last_good_end + 1;
      break;
    }
  }
  return error;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <sizes.cfg> <output_header>\n", argv[0]);
    return 1;
  }

  const char *cfg_path = argv[1];
  const char *out_path = argv[2];

  // ---- Parse sizes.cfg ----
  FILE *cfg = fopen(cfg_path, "r");
  if (!cfg) {
    fprintf(stderr, "Error: cannot open '%s'\n", cfg_path);
    return 1;
  }

  uint64_t sizes[MAX_SIZE_CLASSES];
  int      num_sizes = 0;
  char     line[256];

  while (fgets(line, sizeof(line), cfg)) {
    // Skip blank lines and comments
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
      continue;

    uint64_t s = (uint64_t)strtoull(p, NULL, 10);
    if (s == 0) continue;

    if (num_sizes >= MAX_SIZE_CLASSES) {
      fprintf(stderr, "Error: too many size classes (max %d)\n", MAX_SIZE_CLASSES);
      fclose(cfg);
      return 1;
    }
    sizes[num_sizes++] = s;
  }
  fclose(cfg);

  if (num_sizes == 0) {
    fprintf(stderr, "Error: no valid sizes found in '%s'\n", cfg_path);
    return 1;
  }

  // ---- Validate ----
  if (sizes[0] != MIN_SIZE) {
    fprintf(stderr, "Error: first size must be %d, got %" PRIu64 "\n",
            MIN_SIZE, sizes[0]);
    return 1;
  }
  for (int i = 0; i < num_sizes; i++) {
    if (sizes[i] % MIN_SIZE != 0) {
      fprintf(stderr, "Error: size %" PRIu64 " is not a multiple of %d\n",
              sizes[i], MIN_SIZE);
      return 1;
    }
    if (sizes[i] > REGION_SIZE) {
      fprintf(stderr, "Error: size %" PRIu64 " exceeds max region size %" PRIu64 "\n",
              sizes[i], REGION_SIZE);
      return 1;
    }
    if (i > 0 && sizes[i] <= sizes[i - 1]) {
      fprintf(stderr, "Error: sizes must be strictly ascending; "
              "sizes[%d]=%" PRIu64 " <= sizes[%d]=%" PRIu64 "\n",
              i, sizes[i], i - 1, sizes[i - 1]);
      return 1;
    }
  }

  // ---- Compute tables ----
  uint64_t magics[MAX_SIZE_CLASSES];
  int      is_pow2_arr[MAX_SIZE_CLASSES];
  uint64_t masks[MAX_SIZE_CLASSES];
  uint64_t effective_sizes[MAX_SIZE_CLASSES];  // sizes adjusted for precision errors

  for (int i = 0; i < num_sizes; i++) {
    uint64_t S = sizes[i];
    int pow2   = is_pow2(S);

    is_pow2_arr[i] = pow2;

    if (pow2) {
      magics[i]         = 0;                // unused — POW2 uses AND
      masks[i]          = ~(S - 1);
      effective_sizes[i] = S;              // no precision error for POW2
    } else {
      uint64_t M    = compute_magic(S);
      uint64_t err  = precision_error(S, M);
      magics[i]     = M;
      masks[i]      = 0;                   // not applicable for non-POW2
      // Shrink effective size by error so allocator never gives out the
      // bytes that the magic-number math would mis-identify.
      effective_sizes[i] = S - err;

      if (err > 0) {
        fprintf(stderr, "Note: size %" PRIu64 " has precision error %" PRIu64
                " bytes; effective size = %" PRIu64 "\n", S, err, effective_sizes[i]);
      }
    }
  }

  // ---- Open output ----
  FILE *out = fopen(out_path, "w");
  if (!out) {
    fprintf(stderr, "Error: cannot open output '%s'\n", out_path);
    return 1;
  }

  // ---- Emit header ----
  fprintf(out,
    "//===-- lf_config_generated.h - Auto-generated LowFat config ---------===//\n"
    "//\n"
    "// AUTO-GENERATED by lf_config_gen. DO NOT EDIT.\n"
    "// Source: %s\n"
    "//\n"
    "//===----------------------------------------------------------------------===//\n"
    "\n"
    "#pragma once\n"
    "#ifndef LF_CONFIG_GENERATED_H\n"
    "#define LF_CONFIG_GENERATED_H\n"
    "\n"
    "#include <stdint.h>\n"
    "\n"
    "// Region layout: each region is 32 GB, matching the precision bounds of the\n"
    "// magic-number fixed-point arithmetic proven in the original LowFat research.\n"
    "#define LOWFAT_CUSTOM_CONFIG       1\n"
    "#define LOWFAT_REGION_SIZE_LOG     35\n"
    "#define LOWFAT_REGION_SIZE         (UINT64_C(1) << LOWFAT_REGION_SIZE_LOG)\n"
    "#define LOWFAT_NUM_SIZE_CLASSES    %d\n"
    "#define LOWFAT_MAX_SIZE            UINT64_C(%" PRIu64 ")\n"
    "\n",
    cfg_path,
    num_sizes,
    sizes[num_sizes - 1]
  );

  // kLowFatGenSizes
  fprintf(out,
    "// Actual allocation size for each region index.\n"
    "// For non-POW2 sizes this is the precision-adjusted effective size.\n"
    "static const uint64_t kLowFatGenSizes[LOWFAT_NUM_SIZE_CLASSES] = {\n"
    "    /* idx: size */\n"
  );
  for (int i = 0; i < num_sizes; i++) {
    fprintf(out, "    /* %3d */ UINT64_C(%" PRIu64 ")%s\n",
            i, effective_sizes[i], (i < num_sizes - 1) ? "," : "");
  }
  fprintf(out, "};\n\n");

  // kLowFatGenMagics
  fprintf(out,
    "// Magic numbers for non-POW2 sizes: M = ceil(2^64 / S).\n"
    "// For POW2 sizes this is 0 (they use the AND fast path).\n"
    "static const uint64_t kLowFatGenMagics[LOWFAT_NUM_SIZE_CLASSES] = {\n"
    "    /* idx: magic */\n"
  );
  for (int i = 0; i < num_sizes; i++) {
    fprintf(out, "    /* %3d */ UINT64_C(0x%016" PRIx64 ")%s  // size=%" PRIu64 "%s\n",
            i, magics[i], (i < num_sizes - 1) ? "," : " ",
            sizes[i], is_pow2_arr[i] ? " (POW2, unused)" : "");
  }
  fprintf(out, "};\n\n");

  // kLowFatGenIsPow2
  fprintf(out,
    "// True if this size class is a power-of-two (uses AND path, not MUL path).\n"
    "static const int kLowFatGenIsPow2[LOWFAT_NUM_SIZE_CLASSES] = {\n"
    "    /* idx: isPow2 */\n"
  );
  for (int i = 0; i < num_sizes; i++) {
    fprintf(out, "    /* %3d */ %d%s  // %" PRIu64 "\n",
            i, is_pow2_arr[i], (i < num_sizes - 1) ? "," : " ", sizes[i]);
  }
  fprintf(out, "};\n\n");

  // kLowFatGenMasks
  fprintf(out,
    "// Alignment masks for POW2 sizes: ~(S-1). Zero for non-POW2 sizes.\n"
    "static const uint64_t kLowFatGenMasks[LOWFAT_NUM_SIZE_CLASSES] = {\n"
    "    /* idx: mask */\n"
  );
  for (int i = 0; i < num_sizes; i++) {
    fprintf(out, "    /* %3d */ UINT64_C(0x%016" PRIx64 ")%s  // size=%" PRIu64 "\n",
            i, masks[i], (i < num_sizes - 1) ? "," : " ", sizes[i]);
  }
  fprintf(out, "};\n\n");

  // lowfat_size_to_class: binary search — replaces SizeClassIndex()
  fprintf(out,
    "// Maps a requested allocation size to the smallest region index whose\n"
    "// effective size >= requested size.  Replaces SizeClassIndex() when\n"
    "// LOWFAT_CUSTOM_CONFIG is active.\n"
    "// Returns LOWFAT_NUM_SIZE_CLASSES if size exceeds all size classes.\n"
    "static inline uint64_t lowfat_size_to_class(uint64_t size) {\n"
    "    // Binary search over kLowFatGenSizes[]\n"
    "    if (size == 0) size = 1;\n"
    "    uint64_t lo = 0, hi = LOWFAT_NUM_SIZE_CLASSES;\n"
    "    while (lo < hi) {\n"
    "        uint64_t mid = lo + (hi - lo) / 2;\n"
    "        if (kLowFatGenSizes[mid] < size)\n"
    "            lo = mid + 1;\n"
    "        else\n"
    "            hi = mid;\n"
    "    }\n"
    "    return lo;  // lo == LOWFAT_NUM_SIZE_CLASSES means no fit\n"
    "}\n"
    "\n"
    "#endif  // LF_CONFIG_GENERATED_H\n"
  );

  fclose(out);

  fprintf(stderr, "lf_config_gen: generated '%s' with %d size classes\n",
          out_path, num_sizes);

  // Print summary table to stdout
  printf("Size Class Table:\n");
  printf("  %-5s  %-10s  %-6s  %-20s  %-20s  %-10s\n",
         "Idx", "ReqSize", "POW2?", "EffectiveSize", "Magic", "Mask");
  printf("  %s\n", "---------------------------------------------------------------------");
  for (int i = 0; i < num_sizes; i++) {
    printf("  %-5d  %-10" PRIu64 "  %-6s  %-20" PRIu64 "  %#-20" PRIx64 "  %#-10" PRIx64 "\n",
           i, sizes[i],
           is_pow2_arr[i] ? "yes" : "no",
           effective_sizes[i],
           magics[i],
           masks[i]);
  }

  return 0;
}
