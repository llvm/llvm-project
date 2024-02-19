// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify=kprintf,nonkprintf,expected
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify=kprintf,nonkprintf,expected
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -Wno-format-truncation -Wno-format-overflow %s -verify
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 -Wno-format-truncation -Wno-format-overflow %s -verify
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -Wno-format-truncation-non-kprintf -Wno-format-overflow-non-kprintf %s -verify=kprintf,expected
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 -Wno-format-truncation-non-kprintf -Wno-format-overflow-non-kprintf %s -verify=kprintf,expected
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -Wno-format-overflow -Wno-format-truncation -Wformat-truncation-non-kprintf -Wformat-overflow-non-kprintf %s -verify=nonkprintf,expected
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 -Wno-format-overflow -Wno-format-truncation -Wformat-truncation-non-kprintf -Wformat-overflow-non-kprintf %s -verify=nonkprintf,expected

typedef unsigned long size_t;

#ifdef __cplusplus
extern "C" {
#endif

extern int sprintf(char *str, const char *format, ...);

#ifdef __cplusplus
}
#endif

void call_snprintf(double d, int n, int *ptr) {
  char buf[10];
  __builtin_snprintf(buf, 10, "merp");
  __builtin_snprintf(buf, 11, "merp"); // expected-warning {{'snprintf' size argument is too large; destination buffer has size 10, but size argument is 11}}
  __builtin_snprintf(buf, 12, "%#12x", n); // kprintf-warning {{'snprintf' will always be truncated; specified size is 12, but format string expands to at least 13}} \
                                           // expected-warning {{'snprintf' size argument is too large; destination buffer has size 10, but size argument is 12}}
  __builtin_snprintf(buf, 0, "merp");
  __builtin_snprintf(buf, 3, "merp"); // kprintf-warning {{'snprintf' will always be truncated; specified size is 3, but format string expands to at least 5}}
  __builtin_snprintf(buf, 4, "merp"); // kprintf-warning {{'snprintf' will always be truncated; specified size is 4, but format string expands to at least 5}}
  __builtin_snprintf(buf, 5, "merp");
  __builtin_snprintf(buf, 1, "%.1000g", d); // kprintf-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_snprintf(buf, 5, "%.1000g", d);
  __builtin_snprintf(buf, 5, "%.1000G", d);
  __builtin_snprintf(buf, 10, " %#08x", n);
  __builtin_snprintf(buf, 2, "%#x", n);
  __builtin_snprintf(buf, 2, "%#X", n);
  __builtin_snprintf(buf, 2, "%#o", n);
  __builtin_snprintf(buf, 1, "%#x", n); // kprintf-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_snprintf(buf, 1, "%#X", n); // kprintf-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_snprintf(buf, 1, "%#o", n); // kprintf-warning {{'snprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  char node_name[6];
  __builtin_snprintf(node_name, sizeof(node_name), "%pOFn", ptr); // nonkprintf-warning {{'snprintf' will always be truncated; specified size is 6, but format string expands to at least 7}}
  __builtin_snprintf(node_name, sizeof(node_name), "12345%pOFn", ptr); // nonkprintf-warning {{'snprintf' will always be truncated; specified size is 6, but format string expands to at least 12}}
  __builtin_snprintf(node_name, sizeof(node_name), "123456%pOFn", ptr); // nonkprintf-warning {{'snprintf' will always be truncated; specified size is 6, but format string expands to at least 13}}
}

void call_vsnprintf(void) {
  char buf[10];
  __builtin_va_list list;
  __builtin_vsnprintf(buf, 10, "merp", list);
  __builtin_vsnprintf(buf, 11, "merp", list); // expected-warning {{'vsnprintf' size argument is too large; destination buffer has size 10, but size argument is 11}}
  __builtin_vsnprintf(buf, 0, "merp", list);
  __builtin_vsnprintf(buf, 3, "merp", list); // kprintf-warning {{'vsnprintf' will always be truncated; specified size is 3, but format string expands to at least 5}}
  __builtin_vsnprintf(buf, 4, "merp", list); // kprintf-warning {{'vsnprintf' will always be truncated; specified size is 4, but format string expands to at least 5}}
  __builtin_vsnprintf(buf, 5, "merp", list);
  __builtin_vsnprintf(buf, 1, "%.1000g", list); // kprintf-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_vsnprintf(buf, 5, "%.1000g", list);
  __builtin_vsnprintf(buf, 5, "%.1000G", list);
  __builtin_vsnprintf(buf, 10, " %#08x", list);
  __builtin_vsnprintf(buf, 2, "%#x", list);
  __builtin_vsnprintf(buf, 2, "%#X", list);
  __builtin_vsnprintf(buf, 2, "%#o", list);
  __builtin_vsnprintf(buf, 1, "%#x", list); // kprintf-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_vsnprintf(buf, 1, "%#X", list); // kprintf-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  __builtin_vsnprintf(buf, 1, "%#o", list); // kprintf-warning {{'vsnprintf' will always be truncated; specified size is 1, but format string expands to at least 2}}
  char node_name[6];
  __builtin_snprintf(node_name, sizeof(node_name), "%pOFn", list); // nonkprintf-warning {{'snprintf' will always be truncated; specified size is 6, but format string expands to at least 7}}
  __builtin_snprintf(node_name, sizeof(node_name), "12345%pOFn", list); // nonkprintf-warning {{'snprintf' will always be truncated; specified size is 6, but format string expands to at least 12}}
  __builtin_snprintf(node_name, sizeof(node_name), "123456%pOFn", list); // nonkprintf-warning {{'snprintf' will always be truncated; specified size is 6, but format string expands to at least 13}}
}

void call_sprintf_chk(char *buf) {
  __builtin___sprintf_chk(buf, 1, 6, "hell\n");
  __builtin___sprintf_chk(buf, 1, 5, "hell\n");     // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 5, but format string expands to at least 6}}
  __builtin___sprintf_chk(buf, 1, 6, "hell\0 boy"); // expected-warning {{format string contains '\0' within the string body}}
  __builtin___sprintf_chk(buf, 1, 2, "hell\0 boy"); // expected-warning {{format string contains '\0' within the string body}} \
                                                    // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 2, but format string expands to at least 5}}
  __builtin___sprintf_chk(buf, 1, 6, "hello");
  __builtin___sprintf_chk(buf, 1, 5, "hello"); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 5, but format string expands to at least 6}}
  __builtin___sprintf_chk(buf, 1, 2, "%c", '9');
  __builtin___sprintf_chk(buf, 1, 1, "%c", '9'); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%d", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%d", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%i", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%i", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%o", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%o", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%u", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%u", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%x", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%x", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%X", 9);
  __builtin___sprintf_chk(buf, 1, 1, "%X", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%hhd", (char)9);
  __builtin___sprintf_chk(buf, 1, 1, "%hhd", (char)9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%hd", (short)9);
  __builtin___sprintf_chk(buf, 1, 1, "%hd", (short)9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%ld", 9l);
  __builtin___sprintf_chk(buf, 1, 1, "%ld", 9l); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%lld", 9ll);
  __builtin___sprintf_chk(buf, 1, 1, "%lld", 9ll); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 2, "%%");
  __builtin___sprintf_chk(buf, 1, 1, "%%"); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 1, but format string expands to at least 2}}
  __builtin___sprintf_chk(buf, 1, 4, "%#x", 9);
  __builtin___sprintf_chk(buf, 1, 3, "%#x", 9);
  __builtin___sprintf_chk(buf, 1, 4, "%p", (void *)9);
  __builtin___sprintf_chk(buf, 1, 3, "%p", (void *)9); // nonkprintf-warning {{'sprintf' will always overflow; destination buffer has size 3, but format string expands to at least 4}}
  __builtin___sprintf_chk(buf, 1, 3, "%+d", 9);
  __builtin___sprintf_chk(buf, 1, 2, "%+d", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 2, but format string expands to at least 3}}
  __builtin___sprintf_chk(buf, 1, 3, "% i", 9);
  __builtin___sprintf_chk(buf, 1, 2, "% i", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 2, but format string expands to at least 3}}
  __builtin___sprintf_chk(buf, 1, 6, "%5d", 9);
  __builtin___sprintf_chk(buf, 1, 5, "%5d", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 5, but format string expands to at least 6}}
  __builtin___sprintf_chk(buf, 1, 9, "%f", 9.f);
  __builtin___sprintf_chk(buf, 1, 8, "%f", 9.f); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 8, but format string expands to at least 9}}
  __builtin___sprintf_chk(buf, 1, 9, "%Lf", (long double)9.);
  __builtin___sprintf_chk(buf, 1, 8, "%Lf", (long double)9.); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 8, but format string expands to at least 9}}
  __builtin___sprintf_chk(buf, 1, 10, "%+f", 9.f);
  __builtin___sprintf_chk(buf, 1, 9, "%+f", 9.f); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 9, but format string expands to at least 10}}
  __builtin___sprintf_chk(buf, 1, 12, "%e", 9.f);
  __builtin___sprintf_chk(buf, 1, 11, "%e", 9.f); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 11, but format string expands to at least 12}}
}

void call_sprintf(void) {
  char buf[6];
  sprintf(buf, "hell\0 boy"); // expected-warning {{format string contains '\0' within the string body}}
  sprintf(buf, "hello b\0y"); // expected-warning {{format string contains '\0' within the string body}} \
                              // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 8}}
  sprintf(buf, "hello");
  sprintf(buf, "hello!"); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%%");
  sprintf(buf, "12345%%"); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%c", '9');
  sprintf(buf, "12345%c", '9'); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%d", 9);
  sprintf(buf, "12345%d", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "1234%lld", 9ll);
  sprintf(buf, "12345%lld", 9ll); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "12%#x", 9);
  sprintf(buf, "123%#x", 9);
  sprintf(buf, "12%p", (void *)9);
  sprintf(buf, "123%p", (void *)9); // nonkprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "123%+d", 9);
  sprintf(buf, "1234%+d", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "123% i", 9);
  sprintf(buf, "1234% i", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%5d", 9);
  sprintf(buf, "1%5d", 9); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%.3f", 9.f);
  sprintf(buf, "5%.3f", 9.f); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%+.2f", 9.f);
  sprintf(buf, "%+.3f", 9.f); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
  sprintf(buf, "%.0e", 9.f);
  sprintf(buf, "5%.1e", 9.f); // kprintf-warning {{'sprintf' will always overflow; destination buffer has size 6, but format string expands to at least 8}}
}
