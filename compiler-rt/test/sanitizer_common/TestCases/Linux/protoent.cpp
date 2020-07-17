// RUN: %clangxx -std=c++11 -O0 -g %s -o %t
// RUN: %clangxx -fno-sanitize=all -std=c++11 -O0 -g %s -o %t.nosan
// RUN: diff <(%run %t 2>&1) <(%run %t.nosan 2>&1)
// REQUIRES: !android

#include <assert.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>

void print_protoent(protoent *curr_entry) {
  fprintf(stderr, "%s (%d)\n", curr_entry->p_name, curr_entry->p_proto);

  char **aliases = curr_entry->p_aliases;
  while (char *alias = *aliases++) {
    fprintf(stderr, "  alias %s\n", alias);
  }
}

void print_all_protoent() {
  protoent entry;
  char buf[1024];
  protoent *curr_entry;

  while (getprotoent_r(&entry, buf, sizeof(buf), &curr_entry) != ENOENT && curr_entry) {
    print_protoent(curr_entry);
  }
}

void print_protoent_by_name(const char *name) {
  protoent entry;
  char buf[1024];
  protoent *curr_entry;

  int res = getprotobyname_r(name, &entry, buf, sizeof(buf), &curr_entry);
  assert(!res && curr_entry);
  print_protoent(curr_entry);
}

void print_protoent_by_num(int num) {
  protoent entry;
  char buf[1024];
  protoent *curr_entry;

  int res = getprotobynumber_r(num, &entry, buf, sizeof(buf), &curr_entry);
  assert(!res && curr_entry);
  print_protoent(curr_entry);
}

int main() {
  fprintf(stderr, "All protoent\n");
  print_all_protoent();

  fprintf(stderr, "Protoent by name\n");
  print_protoent_by_name("ipv6");

  fprintf(stderr, "Protoent by num\n");
  print_protoent_by_num(17);
  return 0;
}
