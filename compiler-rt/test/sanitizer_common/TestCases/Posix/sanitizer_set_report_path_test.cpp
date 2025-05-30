// Test __sanitizer_set_report_path and __sanitizer_get_report_path:
// RUN: %clangxx -O2 %s -o %t
// RUN: %env HOME=%t.homedir TMPDIR=%t.tmpdir %run %t 2>%t.err | FileCheck %s
// RUN: FileCheck %s --input-file=%t.err --check-prefix=ERROR

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  char buff[4096];
  sprintf(buff, "%s.report_path/report", argv[0]);
  __sanitizer_set_report_path(buff);
  // CHECK: {{.*}}.report_path/report.[[PID:[0-9]+]]
  printf("%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%H/foo");
  __sanitizer_set_report_path(buff);
  // CHECK: [[T:.*]].homedir/foo.[[PID]]
  printf("%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%t/foo");
  __sanitizer_set_report_path(buff);
  // CHECK: [[T]].tmpdir/foo.[[PID]]
  printf("%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%H/%p/%%foo");
  __sanitizer_set_report_path(buff);
  // CHECK: [[T]].homedir/[[PID]]/%foo.[[PID]]
  printf("%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%%foo%%bar");
  __sanitizer_set_report_path(buff);
  // CHECK: %foo%bar.[[PID]]
  printf("%s\n", __sanitizer_get_report_path());

  strcpy(buff, "%%foo%ba%%r");
  __sanitizer_set_report_path(buff);
  // ERROR: Unexpected pattern: %%foo%ba%%r
  // CHECK: %%foo%ba%%r.[[PID]]
  printf("%s\n", __sanitizer_get_report_path());
}
