/* Measure memcpy performance.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#define MIN_PAGE_SIZE (512*1024+getpagesize())
#define TEST_MAIN
#define TEST_NAME "memcpy"
#include "bench-string.h"
#include <assert.h>
#include "json-lib.h"

#define MAX_COPIES 8192

IMPL (memcpy, 1)

typedef struct { uint16_t size; uint16_t freq; } freq_data_t;
typedef struct { uint8_t align; uint16_t freq; } align_data_t;

#define SIZE_NUM 65536
#define SIZE_MASK (SIZE_NUM-1)
static uint8_t size_arr[SIZE_NUM];

/* Frequency data for memcpy of less than 4096 bytes based on SPEC2017.  */
static freq_data_t size_freq[] =
{
{ 32, 22320}, { 16, 9554}, {  8, 8915}, {152, 5327}, {  4, 2159}, {292, 2035},
{ 12, 1608}, { 24, 1343}, {1152, 895}, {144, 813}, {884, 733}, {284, 721},
{120, 661}, {  2, 649}, {882, 550}, {  5, 475}, {  7, 461}, {108, 460},
{ 10, 361}, {  9, 361}, {  6, 334}, {  3, 326}, {464, 308}, {2048, 303},
{  1, 298}, { 64, 250}, { 11, 197}, {296, 194}, { 68, 187}, { 15, 185},
{192, 184}, {1764, 183}, { 13, 173}, {560, 126}, {160, 115}, {288,  96},
{104,  96}, {1144,  83}, { 18,  80}, { 23,  78}, { 40,  77}, { 19,  68},
{ 48,  63}, { 17,  57}, { 72,  54}, {1280,  51}, { 20,  49}, { 28,  47},
{ 22,  46}, {640,  45}, { 25,  41}, { 14,  40}, { 56,  37}, { 27,  35},
{ 35,  33}, {384,  33}, { 29,  32}, { 80,  30}, {4095,  22}, {232,  22},
{ 36,  19}, {184,  17}, { 21,  17}, {256,  16}, { 44,  15}, { 26,  15},
{ 31,  14}, { 88,  14}, {176,  13}, { 33,  12}, {1024,  12}, {208,  11},
{ 62,  11}, {128,  10}, {704,  10}, {324,  10}, { 96,  10}, { 60,   9},
{136,   9}, {124,   9}, { 34,   8}, { 30,   8}, {480,   8}, {1344,   8},
{273,   7}, {520,   7}, {112,   6}, { 52,   6}, {344,   6}, {336,   6},
{504,   5}, {168,   5}, {424,   5}, {  0,   4}, { 76,   3}, {200,   3},
{512,   3}, {312,   3}, {240,   3}, {960,   3}, {264,   2}, {672,   2},
{ 38,   2}, {328,   2}, { 84,   2}, { 39,   2}, {216,   2}, { 42,   2},
{ 37,   2}, {1608,   2}, { 70,   2}, { 46,   2}, {536,   2}, {280,   1},
{248,   1}, { 47,   1}, {1088,   1}, {1288,   1}, {224,   1}, { 41,   1},
{ 50,   1}, { 49,   1}, {808,   1}, {360,   1}, {440,   1}, { 43,   1},
{ 45,   1}, { 78,   1}, {968,   1}, {392,   1}, { 54,   1}, { 53,   1},
{ 59,   1}, {376,   1}, {664,   1}, { 58,   1}, {272,   1}, { 66,   1},
{2688,   1}, {472,   1}, {568,   1}, {720,   1}, { 51,   1}, { 63,   1},
{ 86,   1}, {496,   1}, {776,   1}, { 57,   1}, {680,   1}, {792,   1},
{122,   1}, {760,   1}, {824,   1}, {552,   1}, { 67,   1}, {456,   1},
{984,   1}, { 74,   1}, {408,   1}, { 75,   1}, { 92,   1}, {576,   1},
{116,   1}, { 65,   1}, {117,   1}, { 82,   1}, {352,   1}, { 55,   1},
{100,   1}, { 90,   1}, {696,   1}, {111,   1}, {880,   1}, { 79,   1},
{488,   1}, { 61,   1}, {114,   1}, { 94,   1}, {1032,   1}, { 98,   1},
{ 87,   1}, {584,   1}, { 85,   1}, {648,   1}, {0, 0}
};

#define ALIGN_NUM 1024
#define ALIGN_MASK (ALIGN_NUM-1)
static uint8_t src_align_arr[ALIGN_NUM];
static uint8_t dst_align_arr[ALIGN_NUM];

/* Source alignment frequency for memcpy based on SPEC2017.  */
static align_data_t src_align_freq[] =
{
  {8, 300}, {16, 292}, {32, 168}, {64, 153}, {4, 79}, {2, 14}, {1, 18}, {0, 0}
};

/* Destination alignment frequency for memcpy based on SPEC2017.  */
static align_data_t dst_align_freq[] =
{
  {8, 265}, {16, 263}, {64, 209}, {32, 174}, {4, 90}, {2, 10}, {1, 13}, {0, 0}
};

typedef struct
{
  uint64_t src : 24;
  uint64_t dst : 24;
  uint64_t len : 16;
} copy_t;

static copy_t copy[MAX_COPIES];

typedef char *(*proto_t) (char *, const char *, size_t);

static void
init_copy_distribution (void)
{
  int i, j, freq, size, n;

  for (n = i = 0; (freq = size_freq[i].freq) != 0; i++)
    for (j = 0, size = size_freq[i].size; j < freq; j++)
      size_arr[n++] = size;
  assert (n == SIZE_NUM);

  for (n = i = 0; (freq = src_align_freq[i].freq) != 0; i++)
    for (j = 0, size = src_align_freq[i].align; j < freq; j++)
      src_align_arr[n++] = size - 1;
  assert (n == ALIGN_NUM);

  for (n = i = 0; (freq = dst_align_freq[i].freq) != 0; i++)
    for (j = 0, size = dst_align_freq[i].align; j < freq; j++)
      dst_align_arr[n++] = size - 1;
  assert (n == ALIGN_NUM);
}


static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, char *dst, char *src,
	     copy_t *copy, size_t n)
{
  timing_t start, stop, cur;
  size_t iters = INNER_LOOP_ITERS_MEDIUM;

  for (int j = 0; j < n; j++)
    CALL (impl, dst + copy[j].dst, src + copy[j].src, copy[j].len);

  TIMING_NOW (start);
  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < n; j++)
      CALL (impl, dst + copy[j].dst, src + copy[j].src, copy[j].len);
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test (json_ctx_t *json_ctx, size_t max_size)
{
  int i;

  memset (buf1, 1, max_size);

  /* Create a random set of copies with the given size and alignment
     distributions.  */
  for (i = 0; i < MAX_COPIES; i++)
    {
      copy[i].dst = (rand () & (max_size - 1));
      copy[i].dst &= ~dst_align_arr[rand () & ALIGN_MASK];
      copy[i].src = (rand () & (max_size - 1));
      copy[i].src &= ~src_align_arr[rand () & ALIGN_MASK];
      copy[i].len = size_arr[rand () & SIZE_MASK];
    }

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "length", (double) max_size);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    do_one_test (json_ctx, impl, (char *) buf2, (char *) buf1, copy, i);

  json_array_end (json_ctx);
  json_element_object_end (json_ctx);
}

int
test_main (void)
{
  json_ctx_t json_ctx;

  test_init ();
  init_copy_distribution ();

  json_init (&json_ctx, 0, stdout);

  json_document_begin (&json_ctx);
  json_attr_string (&json_ctx, "timing_type", TIMING_TYPE);

  json_attr_object_begin (&json_ctx, "functions");
  json_attr_object_begin (&json_ctx, TEST_NAME);
  json_attr_string (&json_ctx, "bench-variant", "random");

  json_array_begin (&json_ctx, "ifuncs");
  FOR_EACH_IMPL (impl, 0)
    json_element_string (&json_ctx, impl->name);
  json_array_end (&json_ctx);

  json_array_begin (&json_ctx, "results");
  for (int i = 4; i <= 512; i = i * 2)
    do_test (&json_ctx, i * 1024);

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>
