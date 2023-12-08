// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,alpha.security.ArrayBoundV2 \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify

// This test is here to check if there is no significant run-time regression
// related to the assume machinery. The analysis should finish in less than 10
// seconds.

// expected-no-diagnostics

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long uint64_t;

int filter_slice_word(int sat_linesize, int sigma, int radius, uint64_t *sat,
                      uint64_t *square_sat, int width, int height,
                      int src_linesize, int dst_linesize, const uint16_t *src,
                      uint16_t *dst, int jobnr, int nb_jobs) {
  const int starty = height * jobnr / nb_jobs;
  const int endy = height * (jobnr + 1) / nb_jobs;

  for (int y = starty; y < endy; y++) {

    int lower_y = y - radius < 0 ? 0 : y - radius;
    int higher_y = y + radius + 1 > height ? height : y + radius + 1;
    int dist_y = higher_y - lower_y;

    for (int x = 0; x < width; x++) {

      int lower_x = x - radius < 0 ? 0 : x - radius;
      int higher_x = x + radius + 1 > width ? width : x + radius + 1;
      int count = dist_y * (higher_x - lower_x);

      // The below hunk caused significant regression in run-time.
#if 1
      uint64_t sum = sat[higher_y * sat_linesize + higher_x] -
                     sat[higher_y * sat_linesize + lower_x] -
                     sat[lower_y * sat_linesize + higher_x] +
                     sat[lower_y * sat_linesize + lower_x];
      uint64_t square_sum = square_sat[higher_y * sat_linesize + higher_x] -
                            square_sat[higher_y * sat_linesize + lower_x] -
                            square_sat[lower_y * sat_linesize + higher_x] +
                            square_sat[lower_y * sat_linesize + lower_x];
      uint64_t mean = sum / count;
      uint64_t var = (square_sum - sum * sum / count) / count;
      dst[y * dst_linesize + x] =
          (sigma * mean + var * src[y * src_linesize + x]) / (sigma + var);
#endif

    }
  }
  return 0;
}
