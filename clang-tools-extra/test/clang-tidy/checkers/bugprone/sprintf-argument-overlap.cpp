// RUN: %check_clang_tidy %s bugprone-sprintf-argument-overlap %t

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

struct st2_t {
  st_t inner;
};

struct st3_t {
  st2_t inner;
};

void first_arg_overlaps() {
  char buf[10];
  sprintf(buf, "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  snprintf(buf, sizeof(buf), "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: the 4th argument in 'snprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  std::snprintf(buf, sizeof(buf), "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: the 4th argument in 'snprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  sprintf(buf+1, "%s", (buf+1));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  sprintf(buf+1, "%s", buf+2);
  sprintf(buf+1, "%s", buf[1]);

  char* c = &buf[0];
  sprintf(c, "%s", c);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  snprintf(c, sizeof(buf), "%s", c);
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: the 4th argument in 'snprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]

  snprintf(c, sizeof(buf), "%s%s", c, c);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: the 4th argument in 'snprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  // CHECK-MESSAGES: :[[@LINE-2]]:39: warning: the 5th argument in 'snprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]

  char buf2[10];
  sprintf(buf, "%s", buf2);
  sprintf(buf, "%s", buf2, buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: the 4th argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]

  st_t st1, st2;
  sprintf(st1.buf, "%s", st1.buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  sprintf(st1.buf, "%s", st1.buf2);
  sprintf(st1.buf, "%s", st2.buf);

  st3_t st3;
  sprintf(st3.inner.inner.buf, "%s", st3.inner.inner.buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  sprintf((st3.inner.inner.buf), "%s", st3.inner.inner.buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]

  st_t* stp;
  sprintf(stp->buf, "%s", stp->buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  sprintf((stp->buf), "%s", stp->buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  stp = &st1;
  sprintf(stp->buf, "%s", st1.buf);

  char bufs[10][10];
  sprintf(bufs[1], "%s", bufs[1]);
  // CHECK-MESSAGES: :[[@LINE-1]]:26:  warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]
  sprintf(bufs[0], "%s", bufs[1]);

  char bufss[10][10][10];
  sprintf(bufss[0][1], "%s", bufss[0][1]);
  // CHECK-MESSAGES: :[[@LINE-1]]:30:  warning: the 3rd argument in 'sprintf' overlaps the 1st argument, which is undefined behavior [bugprone-sprintf-argument-overlap]

  sprintf(bufss[0][0], "%s", bufss[0][1]);

  int i = 0;
  sprintf(bufss[0][++i], "%s", bufss[0][++i]);
}
