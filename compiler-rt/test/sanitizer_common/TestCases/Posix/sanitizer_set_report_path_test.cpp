// Test __sanitizer_set_report_path and __sanitizer_get_report_path:
// RUN: %clangxx -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char buff[4096];
  if (argc == 1) {
    sprintf(buff, "%s.homedir", argv[0]);
    setenv("HOME", buff, 1);
    sprintf(buff, "%s.tmpdir", argv[0]);
    setenv("TMPDIR", buff, 1);
    char *new_argv[] = {argv[0], (char *)"1", nullptr};
    execv(argv[0], new_argv);
    return 1;
  }

  sprintf(buff, "%s.report_path/report", argv[0]);
  __sanitizer_set_report_path(buff);
  // CHECK: {{.*}}.report_path/report.[[PID:[0-9]+]]
  fprintf(stderr, "%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%H/foo");
  __sanitizer_set_report_path(buff);
  // CHECK: [[T:.*]].homedir/foo.[[PID]]
  fprintf(stderr, "%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%t/foo");
  __sanitizer_set_report_path(buff);
  // CHECK: [[T]].tmpdir/foo.[[PID]]
  fprintf(stderr, "%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%H/%p/%%foo");
  __sanitizer_set_report_path(buff);
  // CHECK: [[T]].homedir/[[PID]]/%foo.[[PID]]
  fprintf(stderr, "%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%%foo%%bar");
  __sanitizer_set_report_path(buff);
  // CHECK: %foo%bar.[[PID]]
  fprintf(stderr, "%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%%foo%ba%%r");
  __sanitizer_set_report_path(buff);
  // CHECK: Unexpected pattern: %%foo%ba%%r
  // CHECK: %%foo%ba%%r.[[PID]]
  fprintf(stderr, "%s\n", __sanitizer_get_report_path());
}
