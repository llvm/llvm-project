// RUN: %check_clang_tidy %s bugprone-undefined-sprintf-overlap %t

using size_t = decltype(sizeof(int));

extern "C" int sprintf(char *s, const char *format, ...);
extern "C" int snprintf(char *s, size_t n, const char *format, ...);

namespace std {
  int snprintf(char *s, size_t n, const char *format, ...);
}

struct st_t {
  char buf[10];
  char buf2[10];
};

void first_arg_overlaps() {
  char buf[10];
  sprintf(buf, "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: argument 'buf' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  snprintf(buf, sizeof(buf), "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: argument 'buf' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  std::snprintf(buf, sizeof(buf), "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: argument 'buf' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]

  char* c = &buf[0];
  sprintf(c, "%s", c);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: argument 'c' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  snprintf(c, sizeof(buf), "%s", c);
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: argument 'c' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]

  snprintf(c, sizeof(buf), "%s%s", c, c);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: argument 'c' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]

  char buf2[10];
  sprintf(buf, "%s", buf2);
  sprintf(buf, "%s", buf2, buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: argument 'buf' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]

  st_t st1, st2;
  sprintf(st1.buf, "%s", st1.buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: argument 'st1.buf' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  sprintf(st1.buf, "%s", st1.buf2);
  sprintf(st1.buf, "%s", st2.buf);

  st_t* stp;
  sprintf(stp->buf, "%s", stp->buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: argument 'stp->buf' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  sprintf((stp->buf), "%s", stp->buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: argument 'stp->buf' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  stp = &st1;
  sprintf(stp->buf, "%s", st1.buf);

  char bufs[10][10];
  sprintf(bufs[1], "%s", bufs[1]);
  // CHECK-MESSAGES: :[[@LINE-1]]:26:  warning: argument 'bufs[1]' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  sprintf(bufs[0], "%s", bufs[1]);

  char bufss[10][10][10];
  sprintf(bufss[0][1], "%s", bufss[0][1]);
}
